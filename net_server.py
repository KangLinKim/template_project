import socket
import threading
import json
import time

class GameServer:
    """Simple TCP game server accepting a single client. Host is authoritative.
    - Accepts a single client; second connection attempts are refused and raise RuntimeError.
    - Receives JSON lines of type {"type":"input","input":{...}}
    - send_state(state) will send JSON line {"type":"state","state":...}
    """
    def __init__(self, host='0.0.0.0', port=50007):
        self.host = host
        self.port = int(port)
        self.sock = None
        self.client_sock = None
        self.running = False
        self.lock = threading.Lock()
        self.client_input = None
        self._threads = []

    def start(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(2)
        self.running = True
        t = threading.Thread(target=self._accept_loop, daemon=True)
        t.start()
        self._threads.append(t)

    def _accept_loop(self):
        while self.running:
            try:
                client, addr = self.sock.accept()
            except Exception:
                break

            with self.lock:
                if self.client_sock is None:
                    self.client_sock = client
                    t = threading.Thread(target=self._recv_loop, args=(client,), daemon=True)
                    t.start()
                    self._threads.append(t)
                else:
                    # refuse extra connection and raise
                    try:
                        err = json.dumps({"type":"error","message":"server full"}) + "\n"
                        client.sendall(err.encode())
                    except Exception:
                        pass
                    client.close()
                    raise RuntimeError("Extra client attempted to connect to server port")

    def _recv_loop(self, client):
        f = client.makefile('r')
        try:
            while self.running:
                line = f.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line.strip())
                except Exception:
                    continue
                if msg.get('type') == 'input':
                    with self.lock:
                        self.client_input = msg.get('input')
        finally:
            with self.lock:
                try:
                    client.close()
                except Exception:
                    pass
                self.client_sock = None

    def get_client_input(self):
        with self.lock:
            # return a snapshot (do not clear, allow repeated reads)
            return self.client_input

    def send_state(self, state):
        with self.lock:
            if not self.client_sock:
                return
            try:
                s = json.dumps({"type":"state","state":state}) + "\n"
                self.client_sock.sendall(s.encode())
            except Exception:
                # drop send errors silently; client thread will notice disconnection
                pass

    def stop(self):
        self.running = False
        with self.lock:
            try:
                if self.client_sock:
                    self.client_sock.close()
            except Exception:
                pass
            try:
                if self.sock:
                    self.sock.close()
            except Exception:
                pass
        # threads are daemon; they'll exit


if __name__ == '__main__':
    import time
    s = GameServer(host="172.22.48.1", port=50007)
    s.start()
    print('server started on port 50007')
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        s.stop()
        print('server stopped')
