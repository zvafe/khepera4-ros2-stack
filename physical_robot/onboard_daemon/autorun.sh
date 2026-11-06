#!/bin/sh
case "$1" in
  start|*)
    killall khepera4_tcp_server 2>/dev/null
    /home/root/khepera4_tcp_server > /var/log/khepera4.log 2>&1 &
    killall -9 gst-launch-0.10 gst-launch-1.0 2>/dev/null
    /home/root/khepera4_camera_streamer.sh > /var/log/khepera4_camera.log 2>&1 &
    ;;
  stop)
    killall khepera4_tcp_server 2>/dev/null
    killall -9 gst-launch-0.10 gst-launch-1.0 2>/dev/null
    ;;
esac
exit 0
