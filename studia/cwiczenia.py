def ktynajwiekszy(t, k):#zwraca k-ty najwiekszy element z listy t
    tab = [0] * k
    for element in t:
        i = k-1
        while i > 0:
            if element <= tab[i - 1]:
                if element >= tab[i]:
                    tab[i] = element
                break
            tab[i] = tab[i - 1]
            i = i - 1
        if i == 0:
            tab[0] = element
    return tab[k - 1]

print(ktynajwiekszy([3,6,4,6,4,3,2,5,7,54], 9))