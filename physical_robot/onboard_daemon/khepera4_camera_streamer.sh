#!/bin/sh
# Khepera IV Camera Streaming Daemon
# Configures OMAP3 ISP media controller pipeline and streams MJPEG over TCP port 5001
killall -9 gst-launch-0.10 gst-launch-1.0 2>/dev/null
media-pipes.sh
media-formats.sh 640 480
exec gst-launch-0.10 -e v4l2src device=/dev/video6 ! video/x-raw-yuv,framerate=10/1 ! autoconvert ! jpegenc quality=65 ! multipartmux ! tcpserversink port=5001
