# net_discovery.py
import socket
import json
import time
import ipaddress

DISCOVERY_PORT = 50008
GAME_PORT = 50007
DISCOVERY_MAGIC = "LAN_RACING_GAME"

def get_broadcast_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
    finally:
        s.close()

    network = ipaddress.ip_network(local_ip + "/24", strict=False)
    return str(network.broadcast_address)

def broadcast_server(stop_flag):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    broadcast_ip = get_broadcast_address()

    msg = json.dumps({
        "magic": DISCOVERY_MAGIC,
        "port": GAME_PORT
    }).encode()

    while not stop_flag.is_set():
        try:
            sock.sendto(msg, (broadcast_ip, DISCOVERY_PORT))
        except OSError as e:
            print("[net] broadcast failed:", e)
        time.sleep(1.0)
        
        
def find_server(timeout=3.0):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('', DISCOVERY_PORT))
    sock.settimeout(timeout)

    start = time.time()
    while time.time() - start < timeout:
        try:
            data, addr = sock.recvfrom(1024)
            msg = json.loads(data.decode())
            if msg.get("magic") == DISCOVERY_MAGIC:
                return addr[0], msg["port"]
        except socket.timeout:
            pass

    return None
