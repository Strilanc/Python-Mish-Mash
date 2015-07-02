# coding=utf-8
"""
Quick and dirty and half-done implementation of the axolotl protocol.
"""

__key_pair_counter = 0


def generate_key_pair():
    global __key_pair_counter
    __key_pair_counter += 1
    return ("private", __key_pair_counter), ("public", __key_pair_counter)


def generate_shared_key(A, B):
    assert A[0] == "public"
    assert B[0] == "public"
    return "shared", {A, B}


def hmac_hash(key, X):
    return "hmac", key, X


def hash(X):
    return "hash", X


def key_derivation(master_key, constant_key):
    return ("derived key", master_key, constant_key)


def key_agreement():
    a, A = generate_key_pair()
    b, B = generate_key_pair()
    a0, A0 = generate_key_pair()
    a1, A1 = generate_key_pair()
    b0, B0 = generate_key_pair()
    b1, B1 = generate_key_pair()
    master_key = hash((generate_shared_key(A, B0), generate_shared_key(A0, B), generate_shared_key(A0, B0)))

    alice_state = State(
        root_key=key_derivation(master_key, 0),
        header_key_send=None,
        header_key_recv=key_derivation(master_key, 1),
        next_header_key_send=key_derivation(master_key, 2),
        next_header_key_recv=key_derivation(master_key, 3),
        chain_key_send=None,
        chain_key_recv=key_derivation(master_key, 4),

        identity_key_send=A,
        identity_key_recv=B,

        ratchet_key_send=None,
        ratchet_key_recv=B1,

        message_number_recv=0,
        message_number_send=0,
        previous_message_number_send=0,
        ratchet_flag=True,
        skipped_header_keys_and_message_keys=[])

    bob_state = alice_state.swap_send_recv().but(ratchet_flag=False)

    return alice_state, bob_state


class State:
    def __init__(self,
                 root_key=None,
                 header_key_send=None,
                 header_key_recv=None,
                 next_header_key_send=None,
                 next_header_key_recv=None,
                 chain_key_send=None,
                 chain_key_recv=None,
                 identity_key_send=None,
                 identity_key_recv=None,
                 ratchet_key_send=None,
                 ratchet_key_recv=None,
                 message_number_send=None,
                 message_number_recv=None,
                 previous_message_number_send=None,
                 ratchet_flag=None,
                 skipped_header_keys_and_message_keys=None):
        self.root_key = root_key
        self.header_key_send = header_key_send
        self.header_key_recv = header_key_recv
        self.next_header_key_send = next_header_key_send
        self.next_header_key_recv = next_header_key_recv
        self.chain_key_send = chain_key_send
        self.chain_key_recv = chain_key_recv
        self.identity_key_send = identity_key_send
        self.identity_key_recv = identity_key_recv
        self.ratchet_key_send = ratchet_key_send
        self.ratchet_key_recv = ratchet_key_recv
        self.message_number_send = message_number_send
        self.message_number_recv = message_number_recv
        self.previous_message_number_send = previous_message_number_send
        self.ratchet_flag = ratchet_flag
        self.skipped_header_keys_and_message_keys = skipped_header_keys_and_message_keys

    def swap_send_recv(self):
        return self.but(
            header_key_send=self.header_key_recv,
            header_key_recv=self.header_key_send,
            next_header_key_send=self.next_header_key_recv,
            next_header_key_recv=self.next_header_key_send,
            chain_key_send=self.chain_key_recv,
            chain_key_recv=self.chain_key_send,
            identity_key_send=self.identity_key_recv,
            identity_key_recv=self.identity_key_send,
            ratchet_key_send=self.ratchet_key_recv,
            ratchet_key_recv=self.ratchet_key_send,
            message_number_send=self.message_number_recv,
            message_number_recv=self.message_number_send)

    def but(self,
            root_key=None,
            header_key_send=None,
            header_key_recv=None,
            next_header_key_send=None,
            next_header_key_recv=None,
            chain_key_send=None,
            chain_key_recv=None,
            identity_key_send=None,
            identity_key_recv=None,
            ratchet_key_send=None,
            ratchet_key_recv=None,
            message_number_send=None,
            message_number_recv=None,
            previous_message_number_send=None,
            ratchet_flag=None,
            skipped_header_keys_and_message_keys=None):

        return State(
            root_key or self.root_key,
            header_key_send or self.header_key_send,
            header_key_recv or self.header_key_recv,
            next_header_key_send or self.next_header_key_send,
            next_header_key_recv or self.next_header_key_recv,
            chain_key_send or self.chain_key_send,
            chain_key_recv or self.chain_key_recv,
            identity_key_send or self.identity_key_send,
            identity_key_recv or self.identity_key_recv,
            ratchet_key_send or self.ratchet_key_send,
            ratchet_key_recv or self.ratchet_key_recv,
            message_number_send or self.message_number_send,
            message_number_recv or self.message_number_recv,
            previous_message_number_send or self.previous_message_number_send,
            ratchet_flag or self.ratchet_flag,
            skipped_header_keys_and_message_keys or self.skipped_header_keys_and_message_keys)

    def send(self, plaintext):
        state = self
        if state.ratchet_flag:
            dh = generate_shared_key(state.ratchet_key_send, state.ratchet_key_recv)
            master = hmac_hash(state.root_key, dh)
            state = state.but(
                ratchet_key_send=generate_key_pair(), #wait, what?
                header_key_send=state.next_header_key_send,
                root_key=key_derivation(master, 1),
                next_header_key_send=key_derivation(master, 2),
                chain_key_send=key_derivation(master, 3),
                previous_message_number_send=state.message_number_send,
                message_number_send=0,
                ratchet_flag=False)

        message_key = hmac_hash(state.chain_key_send, "0")
        meta = state.message_number_send, \
               state.previous_message_number_send, \
               state.ratchet_key_send
        message = "message", encrypt(state.header_key_send, meta), encrypt(message_key, plaintext)
        state = state.but(
            message_number_send=state.message_number_send+1,
            chain_key_send=hmac_hash(state.chain_key_send, "1"))
        return state, message

    def recv(self, message):
        # MK  : message key
        # Np  : Purported message number
        # PNp : Purported previous message number
        # CKp : Purported new chain key
        # DHp : Purported new DHr
        # RKp : Purported new root key
        # NHKp, HKp : Purported new header keys
        pass


def encrypt(key, X):
    return "encrypted", key, X


def decrypt(key, X):
    assert X[0] == "encrypted"
    assert X[1] == key
    return X[2]

    # I = np.mat([[1, 0], [0, 1]])
# X = np.mat([[0, 1], [1, 0]])
# Z_90 = np.mat([[1, 0], [0, 1j]])
# Z_45 = np.mat([[1, 0], [0, (1+1j)/math.sqrt(2)]])
# X_90 = np.mat([[1+1j, -1 - 1j], [-1 - 1j, 1 + 1j]])/2
# Y_90 = np.mat([[0.5+0.5j, -0.5-0.5j], [0.5+0.5j, 0.5+0.5j]])
# Y_45 = np.mat([[0.853553 + 0.353553j, -0.353553 - 0.146447j], [0.353553 + 0.146447j, 0.853553 + 0.353553j]])
#
#
# def Rx(t):
#     c, s = math.cos(t/2), math.sin(t/2)
#     return np.mat([[c, -1j * s], [-1j * s, c]])
#
# def Ry(t):
#     c, s = math.cos(t/2), math.sin(t/2)
#     return np.mat([[c, -s], [s, c]])
#
# def Rz(t):
#     return np.mat([[cmath.exp(1j * -t/2), 0], [0, cmath.exp(1j * t/2)]])
#
# def Sx(t):
#     e = cmath.exp(1j*t)
#     return (I*(1 + e) + X*(1 - e))/2
#
# def Sz(t):
#     return np.mat([[1, 0], [0, cmath.exp(1j * t)]])
#
# def adj(m):
#     return np.transpose(np.conjugate(m))
#
# def factorize_abct(m, return_matrices=False):
#     global_phase = cmath.phase(m[0, 0])
#     m_1 = m * cmath.exp(1j * -global_phase)
#     row_phase = cmath.phase(m_1[1, 0])
#     m_2 = Sz(-row_phase) * m_1
#     c, s = m_2[0, 0], m_2[1, 0]
#     rotation = math.atan2(s.real, c.real) * 2
#     m_3 = Ry(-rotation) * m_2
#     column_phase = cmath.phase(m_3[1, 1])
#     # m_4 = m_3 * Sz(-column_phase)
#
#     global_phase += row_phase/2 + column_phase/2
#     if return_matrices:
#         g = cmath.exp(1j * global_phase)
#         return Rz(row_phase), Ry(rotation), Rz(column_phase), np.mat([[g, 0], [0, g]])
#
#     return row_phase, rotation, column_phase, global_phase
#
# def controlify(m):
#     b, y, s, t = factorize_abct(m)
#     A = Rz(b) * Ry(y/2)
#     B = Ry(-y/2)*Rz(-(s+b)/2)
#     C = Rz((s-b)/2)
#     return A, B, C, t
#
# np.set_printoptions(precision=5, suppress=True)
# print np.mat(factorize_abct(Sz(math.pi/2)))/math.pi
# print controlify(Sz(math.pi/2))
#
#
# # a, b, c, t = factorize_abct(Sz(0.5), True)
# # print factorize_abct(Sz(0.25))
# # print a*b*c*t - Sz(0.5)
#
# # a, b, c, t = factorize_abct()
#     # z1 = Rz(row_phase)
#     # y = Ry(rotation)
#     # z2 = Rz(column_phase)
#     # g = cmath.exp(1j * global_phase)
#
#
# # a = adj(Z_90)*Y_45
# # b = adj(Y_45)
# # c = Z_90
# # theta = Z_45*X*Z_45*X
# # print "A"
# # print a
# # print "B"
# # print b
# # print "C"
# # print c
# # print "ABC"
# # print a*b*c
# # print "AXBXC"
# # print a*X*b*X*c*theta
# #
# #
# #
# # # # board = {}
# # # # def moves(i, j):
# # # #     if i == 0 and j == 0:
# # # #         return 1
# # # #     if (i, j) not in board:
# # # #         board[(i, j)] = (sum(moves(k, j) for k in range(i))
# # # #                          + sum(moves(i, k) for k in range(j)))
# # # #     return board[(i, j)]
# # # #
# # # # print moves(7, 7)
# # # #
# # # #
# # # #
# # def rand_uni():
# #     p = random.random() * 3.14159 * 2
# #     t = random.random() * 3.14159 * 2
# #     x = random.random() * 2 - 1
# #     y = (random.random() * 2 - 1) * math.sqrt(1 - x ** 2)
# #     z = math.sqrt(1 - x ** 2 - y ** 2) * random.choice([-1, 1])
# #     c = math.cos(t)
# #     s = math.sin(t)
# #     return (math.cos(p) + 1j * math.sin(p)) * np.mat([
# #         [1j * c + z * s, x * s - 1j * y * s],
# #         [x * s + 1j * y * s, 1j * c - z * s]])
# # #
# # #
# # best_so_far = 10000
# # for i in range(200000):
# #     v = rand_uni()
# #     u = v * X * Z_90 * adj(v)
# #
# #     dif = u - X
# #     s = np.abs(dif[0,0]) + np.abs(dif[1,0]) + np.abs(dif[0,1]) + np.abs(dif[1,1])
# #     if s > best_so_far:
# #         continue
# #     best_so_far = s
# #     print "v"
# #     print v
# #     print "v X sZ v-"
# #     print u
# #     print "dif"
# #     print dif
# #     print "s"
# #     print s
# #
# # print "Done"