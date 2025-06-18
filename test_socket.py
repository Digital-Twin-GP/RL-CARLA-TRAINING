import socket
import json

HOST = '127.0.0.1'
PORT = 65432

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    buffer = ""
    while True:
        data = s.recv(4096)
        if not data:
            break
        buffer += data.decode()
        while '\n' in buffer:
            line, buffer = buffer.split('\n', 1)
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                # Replace the camera_image with a dummy string or summary
                if "camera_image" in obj:
                    if obj["camera_image"]:
                        obj["camera_image"] = f"<base64 image, length={len(obj['camera_image'])}>"
                    else:
                        obj["camera_image"] = "<no image>"
                print(obj)
            except Exception as e:
                print("JSON decode error:", e)
                print("Problematic line:", line)