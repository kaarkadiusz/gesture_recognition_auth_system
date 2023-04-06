from http.server import BaseHTTPRequestHandler, HTTPServer
import base64
import os
import sqlite3
import hashlib

# Funkcja, która służy do autoryzacji
def verification(sequence, true_key, true_salt):
    if len(true_key) == 0 or len(true_salt) == 0:
        return False

    provided_key = hashlib.pbkdf2_hmac('sha256', sequence.encode('utf-8'), true_salt, 100000, dklen=128)

    if provided_key == true_key:
        return True
    else:
        return False

# Sprawdzenie czy istnieje baza danych, a jeśli nie - tworzona jest nowa
if not os.path.isfile("database/users.db"):
    con = sqlite3.connect("database/users.db")
    cur = con.cursor()
    cur.execute("CREATE TABLE users(login, password, salt)")

    salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac('sha256', "[1][2,3][4]".encode('utf-8'), salt, 100000, dklen=128)
    cur.execute("INSERT INTO users VALUES (?, ?, ?)", ['user1', key, salt])

    con.commit()
else:
    con = sqlite3.connect("database/users.db")
    cur = con.cursor()

# Adres serwera
hostname = "localhost"
port = 8000

# Obsługa zapytań
class HTTPAuthServerHandler(BaseHTTPRequestHandler):
    # Zapytanie GET służące do autoryzacji
    def do_GET(self):
        if self.headers.get("Authorization") is not None:
            basic_auth_key = self.headers.get("Authorization")
            login, password = base64.b64decode(basic_auth_key[6:]).decode('ascii').split(':')
            user = cur.execute(
                "SELECT * FROM users WHERE login = ?",
                [login]
            ).fetchone()
            if user is None:
                self.send_response(402)
            else:
                if verification(password, user[1], user[2]):
                    self.send_response(200)
                else:
                    self.send_response(403)
        else:
            self.send_response(300)
        self.send_header("Content-type", "text/html")
        self.end_headers()

    # Zapytanie POST służące do rejestracji
    def do_POST(self):
        if self.headers.get("Authorization") is not None:
            basic_auth_key = self.headers.get("Authorization")
            login, password = base64.b64decode(basic_auth_key[6:]).decode('ascii').split(':')
            user = cur.execute(
                "SELECT * FROM users WHERE login = ?",
                [login]
            ).fetchone()
            if user is not None:
                self.send_response(402)
            else:
                new_salt = os.urandom(32)
                new_key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), new_salt, 100000, dklen=128)
                cur.execute("INSERT INTO users VALUES (?, ?, ?)",
                            [login, new_key, new_salt])
                con.commit()
                self.send_response(200)
        else:
            self.send_response(401)
        self.send_header("Content-type", "text/html")
        self.end_headers()


if __name__ == "__main__":
    # Stworzenie i uruchomienie serwera
    server = HTTPServer((hostname, port), HTTPAuthServerHandler)
    print("Server started http://%s:%s" % (hostname, port))

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()

    server.server_close()
    print("Server stopped.")
