import socket

try:
    print(socket.gethostbyname("api-inference.huggingface.co"))
except Exception as e:
    print("ERROR:", e)