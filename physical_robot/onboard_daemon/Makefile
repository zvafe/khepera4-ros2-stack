# ==============================================================================
# Khepera IV Embedded TCP Server Makefile
# Supports both native compilation on Khepera IV and cross-compilation with Poky
# ==============================================================================

TARGET = khepera4_tcp_server
SRC = khepera4_tcp_server.c

# Native compiler (default)
CC ?= gcc
CFLAGS ?= -O2 -Wall -DHAVE_LIBKHEPERA
LDFLAGS ?= -lkhepera -lm

all: $(TARGET)

$(TARGET): $(SRC)
	@echo "Compiling $(TARGET)..."
	$(CC) $(CFLAGS) $(SRC) $(LDFLAGS) -o $(TARGET)
	@echo "Build complete: $(TARGET)"

# Mock build without libkhepera (for testing socket protocol on PC)
mock: $(SRC)
	gcc -O2 -Wall $(SRC) -lm -o $(TARGET)_mock
	@echo "Mock build complete: $(TARGET)_mock"

# Cross-compilation helper for Poky toolchain
cross:
	@if [ -z "$$POKY_BBLAYERS_ENV" ] && [ -z "$$OECORE_NATIVE_SYSROOT" ]; then \
		echo "[ERROR] Poky environment not sourced!"; \
		echo "Please run: source /opt/poky/1.8/environment-setup-cortexa8hf-vfp-neon-poky-linux-gnueabi"; \
		exit 1; \
	fi
	$$CC $(CFLAGS) $(SRC) $(LDFLAGS) -o $(TARGET)

clean:
	rm -f $(TARGET) $(TARGET)_mock *.o

.PHONY: all mock cross clean
