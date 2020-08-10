import sys
import copy
import os.path


global G
G = [[0],
     [1],
     [1],
     [1, 2],
     [1, 3],
     [1, 2, 3, 4],
     [1, 5],
     [1, 2, 3, 4, 5, 6],
     [1, 3, 5, 7],
     [1, 2, 4, 5, 7, 8],
     [1, 3, 7, 9],
     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
     [1, 5, 7, 11],
     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
     [1, 3, 5, 9, 11, 13],
     [1, 2, 4, 7, 8, 11, 13, 14],
     [1, 3, 5, 7, 9, 11, 13, 15],
     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
     [1, 5, 7, 11, 13, 17],
     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
     [1, 3, 7, 9, 11, 13, 17, 19]]

global donu
donu = [0, 1, 2, 3, 4, 5]

def check(c, k, p, M, d_row):
    t = 1
    for d in range(1, d_row + 1, 1):
        for i in range(0, d, 1):
            rozdiel = [0 for i in range(0, c, 1)]
            for j in range(0, c*k + 2, 1):
                if i != j and d != j:
                    rozdiel[divmod(M[i][j]-M[d][j], c)[1]] = rozdiel[divmod(M[i][j]-M[d][j], c)[1]] + 1
            for j in range(0, c, 1):
                if rozdiel[j] != k:
                    print(d, i, j)
                    return 0
    return 1

def symetrizuj(M, c, k, p, d_row, pivot):
    d = c
    for j in range(1, d_row + 1, 1):
        if j > pivot:
            d = 2
        for i in range(0, c*k + 2, 1):
            if j < i:
                M[i][j] = divmod(M[j][i] + p, d)[1]
    return M


def num(c, arr):
    n = 0
    if c == 6:
        for i in arr:
            n = n*c + donu[i]
        return n
    else:
        for i in arr:
            n = n*c + i
        return n


def num_matrix(c, k, M, d_row):
    a = 0
    if c == 6:
        for j in range(1, d_row + 1, 1):
            for i in range(2, c*k + 2, 1):
                if j < i:
                    a = a * c + donu[M[j][i]]
        return a
    else:
        for j in range(1, d_row + 1, 1):
            for i in range(2, c * k + 2, 1):
                if j < i:
                    a = a * c + M[j][i]
        return a


def denum(c, dlzka, num):
    arr = [0 for i in range(0, dlzka, 1)]
    if c == 6:
        for i in range(dlzka - 1, -1, -1):
            arr[i] = donu[divmod(num, c)[1]]
            num = divmod(num, c)[0]
        return arr
    else:
        for i in range(dlzka - 1, -1, -1):
            num, arr[i] = divmod(num, c)
        return arr


def denum_matrix(c, k, p, num, d_row, pivot):
    m = [0 for i in range(0, c*k + 2, 1)]
    M = [copy.deepcopy(m) for i in range(0, c*k + 2, 1)]
    if c == 6:
        for j in range(d_row, 0, -1):
            for i in range(c * k + 1, 1, -1):
                if j < i:
                    num, b = divmod(num, c)
                    M[j][i] = donu[b]
        return symetrizuj(M, c, k, p, d_row, pivot)
    else:
        for j in range(d_row, 0, -1):
            for i in range(c * k + 1, 1, -1):
                if j < i:
                    num, M[j][i] = divmod(num, c)
        return symetrizuj(M, c, k, p, d_row, pivot)


def bloks(M, c, k, p ,d_row):
    r = [0 for i in range(0, c*k + 2, 1)]
    for i in range(1, d_row):
        for j in range(d_row + 1, c*k + 2, 1):
            r[j] = r[j]*c + M[i][j]
    blk = [0, d_row + 1]
    for i in range(d_row + 1, c*k + 1, 1):
        if r[i] != r[i + 1]:
            blk.append(i + 1)
    blk.append(c*k + 2)
    return blk


def lex_post_bloky(bloky, doplnok):
    b = bloky
    d = doplnok
    for i in range(len(b)-3, -1, -1): # lebo s prvým blokom sa to hrať nemá, ten je daný
        if i == 0:
            return 0
        pivot = 0
        if d[b[i+2]-1]>d[b[i]]:
            for j in range(b[i + 1] - 1, b[i] - 2, - 1):
                if d[j] < d[b[i + 2] - 1]:
                    sufix = d[j:b[i+1]]
                    len_sufix = b[i+1] - j
                    pivot = j
                    break
            Sufix = list()
            for j in range(len(b)-2,i,-1):
                Sufix = Sufix + d[b[j]:b[j+1]]
            len_Sufix = len(Sufix)
            j = 0
            k = 0
            J = 0
            d1 = list()
            d2 = list()
            while j < len_sufix:
                if k < len_Sufix:
                    if Sufix[k]<=sufix[j]:
                        d2.append(Sufix[k])
                        k = k + 1
                    else:
                        d2.append(sufix[j])
                        j = j + 1
                        if j == 1:
                            J = len(d2)
                else:
                    d2 = d2 + sufix[j:]
                    break
            d2 = d2 + Sufix[k:]
            d1 = d2[J:J + len_sufix]
            d2 = d2[:J] + d2[J + len_sufix:]
            return d[0:pivot] + d1 + d2


def lex_post_komb(bloky, doplnok):
    b = bloky
    d = doplnok
    for i in range(len(b)-3, -2, -1):
        pivot = 0
        if i == -1:
            return 0
        if d[b[i+2]-1] > d[b[i]]:
            for j in range(b[i + 1] - 1, b[i] - 2, - 1):
                if d[j] < d[b[i + 2] - 1]:
                    sufix = d[j:b[i+1]]
                    len_sufix = b[i+1] - j
                    pivot = j
                    break
            Sufix = list()
            for j in range(len(b)-2,i,-1):
                Sufix = Sufix + d[b[j]:b[j+1]]
            len_Sufix = len(Sufix)
            j = 0
            k = 0
            J = 0
            d1 = list()
            d2 = list()
            while j < len_sufix:
                if k < len_Sufix:
                    if Sufix[k]<=sufix[j]:
                        d2.append(Sufix[k])
                        k = k + 1
                    else:
                        d2.append(sufix[j])
                        j = j + 1
                        if j == 1:
                            J = len(d2)
                else:
                    d2 = d2 + sufix[j:]
                    break
            d2 = d2 + Sufix[k:]
            d1 = d2[J:J + len_sufix]
            d2 = d2[:J] + d2[J + len_sufix:]
            return d[0:pivot] + d1 + d2


def lex_post(x):
    zac = 0
    for i in range(len(x) - 1, -1, -1):
        if x[i] > x[i - 1]:
            zac = i
            break
    if zac == 0:
        return 0
    else:
        prvy = x[zac - 1]
        for i in range(zac, len(x), 1):
            if x[i] > prvy:
                nahr = i
        nahr_value = x[nahr]
        x[nahr] = prvy
        x[zac - 1] = nahr_value
        y = copy.deepcopy(x)
        for i in range(0, len(x) - zac, 1):
            y[i + zac] = x[len(x) - 1 - i]
        return y


def usp_nor_tvar(M, c, k, p, perm, g, d_row, pivot):
    poradie = [[i for i in range(0, c*k + 2, 1)], [0 for i in range(0, c*k + 2, 1)]]
    nula = perm[0]
    for i in range(1, c*k + 2, 1):
        for j in range(0, d_row + 1, 1):
            if j < i:
                poradie[1][i] = poradie[1][i] * c + divmod((M[perm[j]][perm[i]] - M[nula][perm[i]] - M[perm[j]][nula])*g, c)[1]
            else:
                poradie[1][i] = poradie[1][i] * c
    t = 0
    while t == 0:
        t = 1
        for i in range(0, c*k + 1, 1):
            if poradie[1][i] > poradie[1][i+1]:
                poradie[0][i] = poradie[0][i] ^ poradie[0][i + 1]
                poradie[1][i] = poradie[1][i] ^ poradie[1][i + 1]
                poradie[0][i + 1] = poradie[0][i] ^ poradie[0][i + 1]
                poradie[1][i + 1] = poradie[1][i] ^ poradie[1][i + 1]
                poradie[0][i] = poradie[0][i] ^ poradie[0][i + 1]
                poradie[1][i] = poradie[1][i] ^ poradie[1][i + 1]
                t = 0
    for i in range(0, pivot + 1, 1):
        if poradie[0][i] != i:
            return 0
    for i in range(pivot + 1, d_row + 1, 1):
        if poradie[0][i] > d_row:
            return 0
    N = [[0 for i in range(0, c*k +2, 1)] for j in range(0, c*k + 2, 1)]
    for i in range(0, c*k + 2, 1):
        for j in range(0, c*k + 2, 1):
            if i != j:
                N[i][j] = divmod((M[perm[poradie[0][i]]][perm[poradie[0][j]]] - M[perm[poradie[0][0]]][perm[poradie[0][j]]] - M[perm[poradie[0][i]]][perm[poradie[0][0]]])*g, c)[1]
    return N


def ekvivalencia_s(c, k, p, nu, d_row, matica):
    pivot = 3
    mat = denum_matrix(c, k, p, nu, d_row, d_row)
    N = list()
    for i in range(0, d_row + 1, 1):
        N.append(int(num_matrix(c, k, mat, i)))
    N_per = list()
    for i in range(0, d_row + 1, 1):
        N_per.append(list())
    bloky = [i for i in range(0, pivot + 2, 1)]
    if d_row > pivot:
        bloky.append(d_row + 1)
    per = [i for i in range(0, d_row + 1, 1)]
    dop = [i for i in range(d_row + 1, c*k + 2, 1)]
    while per != 0:
        for g in G[c]:
            perm = per + dop
            O = usp_nor_tvar(mat, c, k, p, perm, g, d_row, pivot)
            if O == 0:
                continue
            o = num_matrix(c, k, O, pivot)
            if o < N[pivot]:
                print("#" + str(o))
                print(g, perm)
                for i in range(0, d_row + 1):
                    print(O[i], mat[i])
                return 0
            if o == N[pivot]:
                N_per[pivot].append([g, per[:pivot + 1]])
                print([g, per[:pivot + 1]])
        per = lex_post_komb(bloky, per)
    xx = open("ZP/P_" + str(c) + "," + str(k) + "/P_na_M" + str(matica) + "riadok" + str(pivot)  + ".txt", "w")
    print(nu)
    for x in N_per[pivot]:
        print(x)
        t = 0
        for i in x[1]:
            if i > pivot:
                t = 1
                continue
        if t == 1:
            continue
        xx.write(str(num(c * k + 2, [x[0]] + x[1])) + "\n")
    print("\n")
    xx.close()
    t = ekvivalencia(c, k, p, nu, d_row, pivot + 1, N, N_per, matica)
    return t


def ekvivalencia(c, k, p, nu, d_row, pivot, N, N_per, matica):
    mat = denum_matrix(c, k, p, nu, d_row, d_row)
    bloky = [i for i in range(0, pivot + 2, 1)]
    if d_row < pivot:
        return 1
    if d_row > pivot:
        bloky.append(d_row + 1)
    for i, [g, per] in enumerate(N_per[pivot - 1]):
        dop = list()
        for j in range(0, c*k + 2, 1):
            if j not in per:
                dop.append(j)
        for j in dop:
            if j < d_row + 1:
                dopl = list()
                for l in dop:
                    if l != j:
                        dopl.append(l)
                perm = per + [j] + dopl
                O = usp_nor_tvar(mat, c, k, p, perm, g, d_row, pivot)
                if O == 0:
                    continue
                o = num_matrix(c, k, O, pivot)
                if o < N[pivot]:
                    print("#" + str(o))
                    print(g, perm)
                    for i in range(0, d_row + 1):
                        print(O[i], mat[i])
                    return 0
                if o == N[pivot]:
                    N_per[pivot].append([g, per + [j]])
    xx = open("ZP/P_" + str(c) + "," + str(k) + "/P_na_M" + str(matica) + "riadok" + str(pivot) + ".txt", "w")
    print(nu)
    for x in N_per[pivot]:
        print(x)
        t = 0
        for i in x[1]:
            if i > pivot:
                t = 1
                continue
        if t == 1:
            continue
        xx.write(str(num(c*k + 2, [x[0]] + x[1])) + "\n")
    print("\n")
    xx.close()
    t = ekvivalencia(c, k, p, nu, d_row, pivot + 1, N, N_per, matica)
    return t


def init(c, k, d_row):
    try:
        os.mkdir("ZP")
    except OSError:
        pass
    try:
        os.mkdir("ZP/P_" + str(c) + "," + str(k))
    except OSError:
        pass
    if divmod(k, 2)[1] == 0:
        p = 0
    else:
        p = int(divmod(c * (c - 1) / 2, c)[1])
    f = open("FN/FN_" + str(c) + "," + str(k) + "/FN_" + str(c) + "," + str(k) + "_riadok_" + str(d_row - 1) + ".txt", "r")
    for i, s in enumerate(f):
        if s[0] == "#":
            continue
        print("START", d_row, i, s)
        nu = int(s)
        mat = denum_matrix(c, k, p, nu, d_row, d_row)
        if check(c, k, p, mat, d_row) != 1:
            print("CH" + s)
            continue
        if ekvivalencia_s(c, k, p, nu, d_row, i) != 1:
            print("E" + s)
            continue
    f.close()


c = 16
for k in range(1, 2, 1):
    init(c, k, c*k + 1)
