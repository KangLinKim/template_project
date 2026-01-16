import socket
import threading
import json

class GameClient:
    """Simple TCP client that sends input dictionaries and receives authoritative state.
    - connect(host,port)
    - send_input(dict)
    - get_state() -> last received state or None
    """
    def __init__(self):
        self.sock = None
        self.lock = threading.Lock()
        self.running = False
        self.last_state = None

    def connect(self, host, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, int(port)))
        self.running = True
        t = threading.Thread(target=self._recv_loop, daemon=True)
        t.start()

    def _recv_loop(self):
        f = self.sock.makefile('r')
        try:
            while self.running:
                line = f.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line.strip())
                except Exception:
                    continue
                if msg.get('type') == 'state':
                    with self.lock:
                        self.last_state = msg.get('state')
        finally:
            try:
                self.sock.close()
            except Exception:
                pass
            self.running = False

    def send_input(self, inp):
        if not self.sock:
            return
        try:
            s = json.dumps({"type":"input","input":inp}) + "\n"
            self.sock.sendall(s.encode())
        except Exception:
            pass

    def get_state(self):
        with self.lock:
            return self.last_state

    def stop(self):
        self.running = False
        try:
            if self.sock:
                self.sock.close()
        except Exception:
            pass

if __name__ == '__main__':
    c = GameClient()
    c.connect('172.22.48.1', 50007)
    import time
    try:
        while True:
            s = c.get_state()
            if s:
                print('state', s)
            time.sleep(0.5)
    except KeyboardInterrupt:
        c.stop()
