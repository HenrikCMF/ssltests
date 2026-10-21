"""Server view of a client-enforced Gaussian aggregate.

Each coordinate of the aggregate is one draw y = μ + zσ, the same function
as ``DistributionAggregator._distribution_chunk`` (population standard
deviation, divisor K, inverse-CDF z with probability ``tail`` cut from each
side).

The server is a relay. Client-to-client payloads are sealed to the
recipient. Clients reconstruct the sum and the sum of squares on those
channels, draw each z from a joint seed, and open only y. The transcript
printed at the end is every plaintext vector the server receives.

Clients learn μ, σ, and z. The server does not. This file is a reproducible
view simulation of that relay, not a production MPC library: a server that
can read client memory, or a relay that is given the shares in the clear,
can rebuild every input.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np

# Prime field for additive shares. All secret integers live in [0, P).
P = 2**61 - 1
# Decimal fixed point so the demo inputs (0.03, 100, ...) survive quantization.
SCALE = 10**6
TAIL = 0.35
MASTER = b"gauss-secagg-demo-v1"

# Rows are clients, columns are coordinates. Column stories:
# 0: one client carries the value, the other four are exact zeros.
# 1: constant ±C decoy around a small imprint.
# 2: every client sent the same value, so σ = 0 and y = μ.
# 3: ordinary close values.
CLIENTS = (
    (0.03, 0.03, 1.0, 0.50),
    (0.00, 100.0, 1.0, 0.40),
    (0.00, 100.0, 1.0, 0.60),
    (0.00, -100.0, 1.0, 0.55),
    (0.00, -100.0, 1.0, 0.45),
)
COORD_NAMES = ("1-sparse", "pm-C decoy", "agreement", "close")


def to_fp(value: float) -> int:
    """Round a real number into fixed point, as a field element."""
    return int(round(value * SCALE)) % P


def center(value: int) -> int:
    """Lift a field element back to a signed integer near zero."""
    value %= P
    if value > P // 2:
        value -= P
    return value


def from_fp(value: int) -> float:
    return center(value) / SCALE


def square_fp(value: int) -> int:
    """Fixed-point square of a field element. Exact for this demo's inputs."""
    signed = center(value)
    prod = signed * signed
    half = SCALE // 2
    quant = (prod + half) // SCALE
    return quant % P


def erfinv(y: float) -> float:
    """Inverse error function. Newton on math.erf, accurate on the tail band."""
    if y == 0.0:
        return 0.0
    # |2u-1| ≤ 1-2*tail = 0.3 at the default tail, so a zero start converges.
    x = 0.0
    scale = 2.0 / math.sqrt(math.pi)
    for _ in range(12):
        deriv = scale * math.exp(-x * x)
        x -= (math.erf(x) - y) / deriv
    return x


def truncated_z(uniforms: np.ndarray, tail: float) -> np.ndarray:
    """Map U(0, 1) draws to N(0, 1) truncated to probability [tail, 1-tail]."""
    out = np.empty(uniforms.shape[0], dtype=np.float64)
    width = 1.0 - 2.0 * tail
    sqrt2 = math.sqrt(2.0)
    for i, raw in enumerate(uniforms):
        u = float(raw) * width + tail
        out[i] = sqrt2 * erfinv(2.0 * u - 1.0)
    return out


def _seal(dst_key: bytes, nonce: bytes, payload: bytes) -> bytes:
    stream = hashlib.shake_256(dst_key + nonce).digest(len(payload))
    body = bytes(a ^ b for a, b in zip(payload, stream))
    return nonce + body


def _open(dst_key: bytes, blob: bytes) -> bytes:
    nonce, body = blob[:16], blob[16:]
    stream = hashlib.shake_256(dst_key + nonce).digest(len(body))
    return bytes(a ^ b for a, b in zip(body, stream))


def pack_ints(values: list[int]) -> bytes:
    return b"".join(int(v % P).to_bytes(8, "little") for v in values)


def unpack_ints(payload: bytes) -> list[int]:
    return [int.from_bytes(payload[i : i + 8], "little") for i in range(0, len(payload), 8)]


@dataclass
class RelayMessage:
    src: int
    dst: int
    label: str
    blob: bytes


@dataclass
class Server:
    """Relay mailbox. Plaintext delivery accepts only the opened draw."""

    relayed: list[RelayMessage] = field(default_factory=list)
    opened: list[tuple[str, tuple[float, ...]]] = field(default_factory=list)

    def relay(self, src: int, dst: int, label: str, blob: bytes) -> None:
        self.relayed.append(RelayMessage(src, dst, label, blob))

    def deliver_plaintext(self, name: str, vector: list[float]) -> None:
        if name != "y":
            raise RuntimeError(f"server was offered plaintext {name!r}; only y may be opened")
        self.opened.append((name, tuple(vector)))

    def plaintext_vectors(self) -> list[tuple[str, tuple[float, ...]]]:
        return list(self.opened)


def client_key(cid: int) -> bytes:
    return hashlib.sha256(b"key|" + MASTER + cid.to_bytes(2, "little")).digest()


def client_seed(cid: int) -> bytes:
    return hashlib.sha256(b"seed|" + MASTER + cid.to_bytes(2, "little")).digest()


def nonce_for(label: str, src: int, dst: int) -> bytes:
    raw = f"{label}|{src}|{dst}".encode()
    return hashlib.sha256(MASTER + raw).digest()[:16]


def split_secret(secret: list[int], k: int, rng: np.random.Generator) -> list[list[int]]:
    """Additive shares. shares[j] is for client j, and the shares sum to secret."""
    dim = len(secret)
    shares = [[0] * dim for _ in range(k)]
    for coord in range(dim):
        running = 0
        for holder in range(k - 1):
            part = int(rng.integers(0, P))
            shares[holder][coord] = part
            running = (running + part) % P
        shares[k - 1][coord] = (secret[coord] - running) % P
    return shares


def joint_uniforms(seeds: list[bytes], dim: int) -> np.ndarray:
    """One U(0, 1) per coordinate from the concatenation of every client seed."""
    digest = hashlib.sha256(b"".join(seeds)).digest()
    words = [int.from_bytes(digest[i : i + 4], "little") for i in range(0, 32, 4)]
    return np.random.default_rng(words).random(dim)


@dataclass
class Client:
    cid: int
    values: list[int]
    squares: list[int]
    key: bytes
    # Shares of Σx and Σx² held by this client, filled as mail arrives.
    sum_share: list[int] | None = None
    sumsq_share: list[int] | None = None
    inbox: list[RelayMessage] = field(default_factory=list)

    def receive(self, message: RelayMessage) -> None:
        self.inbox.append(message)


def _send(server: Server, clients: list[Client], src: int, dst: int, label: str, payload: bytes) -> None:
    blob = _seal(clients[dst].key, nonce_for(label, src, dst), payload)
    server.relay(src, dst, label, blob)
    clients[dst].receive(server.relayed[-1])


def _take(clients: list[Client], me: int, label: str) -> list[list[int]]:
    found = []
    for message in clients[me].inbox:
        if message.dst != me or message.label != label:
            continue
        raw = _open(clients[me].key, message.blob)
        found.append(unpack_ints(raw))
    return found


def aggregate(server: Server) -> list[float]:
    """Run the sealed protocol. Return the y that every client computed."""
    k = len(CLIENTS)
    dim = len(CLIENTS[0])
    rng = np.random.default_rng(20261002)
    clients = [
        Client(
            cid=i,
            values=[to_fp(v) for v in row],
            squares=[square_fp(to_fp(v)) for v in row],
            key=client_key(i),
        )
        for i, row in enumerate(CLIENTS)
    ]

    for client in clients:
        for label, secret in (("share-x", client.values), ("share-x2", client.squares)):
            parts = split_secret(secret, k, rng)
            for holder, part in enumerate(parts):
                if holder == client.cid:
                    continue
                _send(server, clients, client.cid, holder, label, pack_ints(part))
            # Keep the local share in the inbox as an already-open self message
            # so reconstruction uses one path. Self-mail is not relayed.
            client.inbox.append(
                RelayMessage(client.cid, client.cid, label, _seal(client.key, nonce_for(label, client.cid, client.cid), pack_ints(parts[client.cid])))
            )

    for client in clients:
        dim_x = _take(clients, client.cid, "share-x")
        dim_sq = _take(clients, client.cid, "share-x2")
        if len(dim_x) != k or len(dim_sq) != k:
            raise RuntimeError("a client is missing an input share")
        client.sum_share = [sum(part[c] for part in dim_x) % P for c in range(dim)]
        client.sumsq_share = [sum(part[c] for part in dim_sq) % P for c in range(dim)]

    for client in clients:
        assert client.sum_share is not None and client.sumsq_share is not None
        for holder in range(k):
            if holder == client.cid:
                continue
            _send(server, clients, client.cid, holder, "share-sum", pack_ints(client.sum_share))
            _send(server, clients, client.cid, holder, "share-sumsq", pack_ints(client.sumsq_share))
            _send(server, clients, client.cid, holder, "seed", client_seed(client.cid))
        client.inbox.append(
            RelayMessage(
                client.cid,
                client.cid,
                "share-sum",
                _seal(client.key, nonce_for("share-sum", client.cid, client.cid), pack_ints(client.sum_share)),
            )
        )
        client.inbox.append(
            RelayMessage(
                client.cid,
                client.cid,
                "share-sumsq",
                _seal(client.key, nonce_for("share-sumsq", client.cid, client.cid), pack_ints(client.sumsq_share)),
            )
        )

    draws = []
    for client in clients:
        sum_parts = _take(clients, client.cid, "share-sum")
        sq_parts = _take(clients, client.cid, "share-sumsq")
        total = [sum(part[c] for part in sum_parts) % P for c in range(dim)]
        total_sq = [sum(part[c] for part in sq_parts) % P for c in range(dim)]
        mu = [center(total[c]) / (k * SCALE) for c in range(dim)]
        var = []
        sigma = []
        for c in range(dim):
            second = center(total_sq[c]) / (k * SCALE)
            var_c = second - mu[c] * mu[c]
            var.append(var_c)
            sigma.append(math.sqrt(var_c) if var_c > 0.0 else 0.0)
        seed_msgs = [m for m in client.inbox if m.dst == client.cid and m.label == "seed"]
        seeds = [client_seed(client.cid)] + [_open(client.key, m.blob) for m in seed_msgs]
        seeds_sorted = sorted(seeds)
        z = truncated_z(joint_uniforms(seeds_sorted, dim), TAIL)
        y = [mu[c] + float(z[c]) * sigma[c] for c in range(dim)]
        draws.append(y)
        server.deliver_plaintext("y", y)

    first = draws[0]
    for other in draws[1:]:
        if any(abs(a - b) > 0.0 for a, b in zip(first, other)):
            raise RuntimeError("clients computed different draws")
    return first


def reference_draw(y_from_protocol: list[float]) -> dict[str, list[float] | np.ndarray]:
    """Same μ, σ, and joint z, computed in the clear for the auditor."""
    k = len(CLIENTS)
    dim = len(CLIENTS[0])
    quantized = np.array([[from_fp(to_fp(v)) for v in row] for row in CLIENTS], dtype=np.float64)
    mu = quantized.mean(axis=0)
    sigma = np.sqrt(((quantized - mu) ** 2).mean(axis=0))
    seeds = sorted(client_seed(i) for i in range(k))
    z = truncated_z(joint_uniforms(seeds, dim), TAIL)
    y = mu + z * sigma
    if np.max(np.abs(y - np.array(y_from_protocol))) > 1e-12:
        raise RuntimeError("opened y does not match μ + zσ")
    return {"x": quantized, "mu": mu, "sigma": sigma, "z": z, "y": y}


def _fmt(values) -> str:
    return "[" + ", ".join(f"{float(v): .10f}" for v in values) + "]"


def _report(server: Server, ref: dict) -> None:
    k = len(CLIENTS)
    labels: dict[str, int] = {}
    for message in server.relayed:
        labels[message.label] = labels.get(message.label, 0) + 1
    opened_names = [name for name, _ in server.opened]
    if opened_names != ["y"] * k:
        raise RuntimeError(f"unexpected plaintext deliveries: {opened_names}")
    vectors = {tuple(vec) for _, vec in server.opened}
    if len(vectors) != 1:
        raise RuntimeError("clients opened inconsistent y vectors")

    print("Gaussian secure aggregate, server view")
    print(f"K={k}  D={len(COORD_NAMES)}  tail={TAIL}  fixed point 1/{SCALE}")
    print("Client-to-client payloads are sealed. The server relays them and")
    print("receives y in the clear. μ, σ, and z stay on the client side.")
    print()
    print("Opened to the server")
    print(f"  plaintext vectors : {len(server.opened)} copies of y, one from each client")
    print(f"  y = {_fmt(server.opened[0][1])}")
    print()
    print("Relay traffic with no plaintext vector")
    print(f"  sealed messages   : {len(server.relayed)}")
    for label in ("share-x", "share-x2", "share-sum", "share-sumsq", "seed"):
        print(f"  {label:<14}  {labels.get(label, 0)}")
    sample = server.relayed[0]
    print(
        f"  example           : client {sample.src} -> client {sample.dst}  "
        f"{sample.label}  {len(sample.blob)} bytes  "
        f"hex={sample.blob[:12].hex()}"
    )
    # The recipient key is client state. The server object never has it.
    # One opened share is a one-time pad: it is not that client's update.
    share = unpack_ints(_open(client_key(sample.dst), sample.blob))
    bogus = unpack_ints(_open(b"server-has-no-recipient-key", sample.blob))
    if share == bogus:
        raise RuntimeError("a key the server can hold decrypted a share")
    print(f"  one share inside  : {_fmt([from_fp(v) for v in share])}")
    print(f"  opened without key: {_fmt([from_fp(v) for v in bogus])}")
    print("  local share of each input is not relayed, so the sealed")
    print("  traffic does not determine any client vector")
    print()
    print("Auditor, not delivered")
    print("  columns           : " + ", ".join(COORD_NAMES))
    x = ref["x"]
    for i, row in enumerate(x):
        print(f"  client {i}          : {_fmt(row)}")
    print(f"  μ                 : {_fmt(ref['mu'])}")
    print(f"  σ                 : {_fmt(ref['sigma'])}")
    print(f"  z                 : {_fmt(ref['z'])}")
    print(f"  μ + zσ            : {_fmt(ref['y'])}")
    # Ordinary sum-SecAgg opens Kμ (equivalently μ). Printed so the two
    # opened objects can be compared; it is not in the server transcript.
    print(f"  sum Kμ            : {_fmt(k * ref['mu'])}")
    print()
    print("Per coordinate")
    for c, name in enumerate(COORD_NAMES):
        y_c = float(ref["y"][c])
        mu_c = float(ref["mu"][c])
        sig_c = float(ref["sigma"][c])
        print(
            f"  {name:<12}  y={y_c: .10f}  μ={mu_c: .10f}  σ={sig_c: .10f}  "
            f"z={float(ref['z'][c]): .6f}"
        )
    agree = abs(float(ref["y"][2]) - float(ref["mu"][2]))
    if agree > 1e-15:
        raise RuntimeError("agreement coordinate should open μ, because σ = 0")
    if abs(float(ref["sigma"][0]) - 0.012) > 1e-12:
        raise RuntimeError("1-sparse σ should be 0.4 * |v| = 0.012")
    if abs(float(ref["y"][0]) - float(ref["mu"][0])) < 1e-6:
        raise RuntimeError("1-sparse draw landed on μ; pick a different demo seed")


def main() -> None:
    server = Server()
    y = aggregate(server)
    ref = reference_draw(y)
    _report(server, ref)
    # The server object has no client keys and no client values.
    if hasattr(server, "key") or hasattr(server, "values"):
        raise RuntimeError("server is holding client state")
    print()
    print("check: server plaintext is y, and it matches μ + zσ")


if __name__ == "__main__":
    main()
