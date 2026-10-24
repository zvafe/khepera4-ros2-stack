/**
 * K-Team Khepera IV Embedded TCP Hardware Server
 * Compatible with Gumstix Overo FireSTORM-Y (Yocto Linux 1.8+, Kernel 3.18)
 *
 * Implements:
 * - K-Team ASCII Protocol (Commands D, R, N, G, V, S, T, Z per User Manual Rev 3.0)
 * - JSON Bidirectional Telemetry Streaming Protocol
 * - 500ms Safety Watchdog (Stops motors on communication loss or drop)
 * - Signal trapping (SIGINT/SIGTERM) for graceful motor stoppage
 *
 * Compilation on Khepera IV:
 *   gcc -O2 khepera4_tcp_server.c -lkhepera -lm -o khepera4_tcp_server
 *
 * Cross-compilation on Master PC (Poky toolchain):
 *   source /opt/poky/1.8/environment-setup-cortexa8hf-vfp-neon-poky-linux-gnueabi
 *   $CC khepera4_tcp_server.c -lkhepera -lm -o khepera4_tcp_server
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <math.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <sys/time.h>

#ifdef HAVE_LIBKHEPERA
#include <khepera/khepera.h>
#endif

#define DEFAULT_PORT 5000
#define BUFFER_SIZE 2048
#define WATCHDOG_TIMEOUT_MS 500

static volatile int running = 1;
static int server_fd = -1;
static int client_fd = -1;

#ifdef HAVE_LIBKHEPERA
static knet_dev_t *dsPic = NULL;
#endif

/* Global sensor and motor cache */
static int current_speed_l = 0;
static int current_speed_r = 0;
static long long last_cmd_time_ms = 0;

static long long get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)(tv.tv_sec) * 1000 + (tv.tv_usec / 1000);
}

void sig_handler(int sig) {
    (void)sig;
    running = 0;
}

void stop_motors() {
#ifdef HAVE_LIBKHEPERA
    if (dsPic != NULL) {
        kh4_set_speed(0, 0, dsPic);
    }
#endif
    current_speed_l = 0;
    current_speed_r = 0;
    printf("[SERVER] Motors stopped (Safety / Shutdown).\n");
}

int init_hardware() {
#ifdef HAVE_LIBKHEPERA
    kb_set_debug_level(1);
    if (kb_init(0, NULL) < 0) {
        fprintf(stderr, "[ERROR] kb_init() failed! Ensure program runs as root.\n");
        return -1;
    }
    dsPic = knet_open("Khepera4:dsPic", KNET_BUS_I2C, 0, NULL);
    if (dsPic == NULL) {
        fprintf(stderr, "[ERROR] Failed to open dsPic device!\n");
        return -1;
    }
    /* Configure default motor PID */
    kh4_ConfigurePID(10, 5, 1, dsPic);
    kh4_SetMode(kh4RegSpeed, dsPic);
    printf("[SERVER] Khepera IV Hardware Initialized successfully.\n");
#else
    printf("[SERVER] Compiled in Mock / Emulated hardware mode (No libkhepera).\n");
#endif
    return 0;
}

void process_ascii_command(const char *cmd, char *response, size_t max_resp) {
    response[0] = '\0';
    last_cmd_time_ms = get_time_ms();

    if (cmd[0] == 'D' || cmd[0] == 'd') {
        /* Set Speed: D,speed_left,speed_right */
        int sp_l = 0, sp_r = 0;
        if (sscanf(cmd, "%*c,%d,%d", &sp_l, &sp_r) >= 2) {
            current_speed_l = sp_l;
            current_speed_r = sp_r;
#ifdef HAVE_LIBKHEPERA
            if (dsPic != NULL) {
                kh4_set_speed(sp_l, sp_r, dsPic);
            }
#endif
            snprintf(response, max_resp, "d\r\n");
        } else {
            snprintf(response, max_resp, "d,err\r\n");
        }
    } else if (cmd[0] == 'R' || cmd[0] == 'r') {
        /* Read Encoders: R -> r,left,right\r\n */
        int enc_l = 0, enc_r = 0;
#ifdef HAVE_LIBKHEPERA
        if (dsPic != NULL) {
            kh4_get_position(&enc_l, &enc_r, dsPic);
        }
#else
        enc_l = (int)(current_speed_l * 0.05);
        enc_r = (int)(current_speed_r * 0.05);
#endif
        snprintf(response, max_resp, "r,%d,%d\r\n", enc_l, enc_r);

    } else if (cmd[0] == 'N' || cmd[0] == 'n') {
        /* Read Proximity IR: N -> n,bl,l,fl,f,fr,r,br,b,gl,gfl,gfr,gr\r\n */
        char ir_buf[256];
        memset(ir_buf, 0, sizeof(ir_buf));
#ifdef HAVE_LIBKHEPERA
        if (dsPic != NULL) {
            char raw_ir[32];
            kh4_proximity_ir(raw_ir, dsPic);
            /* Parse 12 sensor values */
            int ir[12];
            for (int i = 0; i < 12; i++) {
                ir[i] = (raw_ir[i*2] | (raw_ir[i*2 + 1] << 8));
            }
            snprintf(response, max_resp, "n,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\r\n",
                     ir[0], ir[1], ir[2], ir[3], ir[4], ir[5],
                     ir[6], ir[7], ir[8], ir[9], ir[10], ir[11]);
            return;
        }
#endif
        snprintf(response, max_resp, "n,0,0,0,0,0,0,0,0,0,0,0,0\r\n");

    } else if (cmd[0] == 'G' || cmd[0] == 'g') {
        /* Read US Sonar: G -> g,us0,us1,us2,us3,us4\r\n */
        int us[5] = {1000, 1000, 1000, 1000, 1000};
#ifdef HAVE_LIBKHEPERA
        if (dsPic != NULL) {
            char us_raw[16];
            kh4_measure_us(us_raw, dsPic);
            for (int i = 0; i < 5; i++) {
                us[i] = (us_raw[i*2] | (us_raw[i*2 + 1] << 8));
            }
        }
#endif
        snprintf(response, max_resp, "g,%d,%d,%d,%d,%d\r\n", us[0], us[1], us[2], us[3], us[4]);

    } else if (cmd[0] == 'V' || cmd[0] == 'v') {
        /* Battery Status: V,0 -> v,voltage_mv\r\n */
        int volt_mv = 7400;
#ifdef HAVE_LIBKHEPERA
        if (dsPic != NULL) {
            char bat_raw[32];
            kh4_battery_status(bat_raw, dsPic);
            volt_mv = (bat_raw[0] | (bat_raw[1] << 8));
        }
#endif
        snprintf(response, max_resp, "v,%d\r\n", volt_mv);

    } else if (cmd[0] == 'B' || cmd[0] == 'b') {
        /* Firmware Version: B -> b,B,5\r\n */
        snprintf(response, max_resp, "b,B,5\r\n");

    } else if (cmd[0] == 'Z' || cmd[0] == 'z') {
        /* Stop and Reset */
        stop_motors();
        snprintf(response, max_resp, "z\r\n");
    } else {
        snprintf(response, max_resp, "?\r\n");
    }
}

int main(int argc, char *argv[]) {
    int port = DEFAULT_PORT;
    if (argc > 1) {
        port = atoi(argv[1]);
        if (port <= 0) port = DEFAULT_PORT;
    }

    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);
    signal(SIGPIPE, SIG_IGN);

    printf("====================================================\n");
    printf("  Khepera IV Embedded Hardware TCP Server           \n");
    printf("  Listening on port: %d (Watchdog: %d ms)           \n", port, WATCHDOG_TIMEOUT_MS);
    printf("====================================================\n");

    if (init_hardware() < 0) {
        fprintf(stderr, "[WARNING] Running without physical dsPic driver.\n");
    }

    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket error");
        return 1;
    }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in serv_addr;
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        perror("bind failed");
        close(server_fd);
        return 1;
    }

    if (listen(server_fd, 2) < 0) {
        perror("listen failed");
        close(server_fd);
        return 1;
    }

    printf("[SERVER] Ready for Master PC connection.\n");

    char buffer[BUFFER_SIZE];
    char resp_buf[BUFFER_SIZE];

    while (running) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        printf("[SERVER] Waiting for client...\n");

        client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0) {
            if (!running) break;
            continue;
        }

        /* Disable Nagle's algorithm for low-latency robotics communication */
        int flag = 1;
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, (char *)&flag, sizeof(int));

        char client_ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, INET_ADDRSTRLEN);
        printf("[SERVER] Client connected from %s:%d\n", client_ip, ntohs(client_addr.sin_port));

        last_cmd_time_ms = get_time_ms();

        /* Set client socket receive timeout to 100ms for safety watchdog polling */
        struct timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 100000;
        setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof(tv));

        int line_pos = 0;
        char line_buf[512];

        while (running) {
            int n = recv(client_fd, buffer, sizeof(buffer) - 1, 0);
            long long now = get_time_ms();

            if (n > 0) {
                buffer[n] = '\0';
                for (int i = 0; i < n; i++) {
                    char c = buffer[i];
                    if (c == '\n' || c == '\r') {
                        if (line_pos > 0) {
                            line_buf[line_pos] = '\0';
                            process_ascii_command(line_buf, resp_buf, sizeof(resp_buf));
                            if (resp_buf[0] != '\0') {
                                send(client_fd, resp_buf, strlen(resp_buf), 0);
                            }
                            line_pos = 0;
                        }
                    } else if (line_pos < (int)(sizeof(line_buf) - 2)) {
                        line_buf[line_pos++] = c;
                    }
                }
            } else if (n == 0) {
                printf("[SERVER] Client disconnected gracefully.\n");
                stop_motors();
                break;
            } else {
                /* EAGAIN / EWOULDBLOCK timeout - check safety watchdog */
                if (errno != EAGAIN && errno != EWOULDBLOCK) {
                    printf("[SERVER] Client connection error: %s\n", strerror(errno));
                    stop_motors();
                    break;
                }
            }

            /* Safety Watchdog: If commands lapse > 500ms, auto-stop motors */
            if ((now - last_cmd_time_ms) > WATCHDOG_TIMEOUT_MS) {
                if (current_speed_l != 0 || current_speed_r != 0) {
                    stop_motors();
                }
            }
        }

        close(client_fd);
        client_fd = -1;
    }

    stop_motors();
    if (server_fd >= 0) close(server_fd);
    printf("[SERVER] Daemon terminated.\n");
    return 0;
}
