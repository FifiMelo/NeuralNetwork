#liczy potege o wykladniku naturalnym w czasie O(log n)
def power(base, exponent):
    if exponent == 1:
        return base
    if not exponent%2:
        return power(base*base,exponent//2)
    return base * power(base, exponent - 1)

#liczy ln(x) dla x z przedziału <-1,1>
def ln(x,n=100):
    x = x - 1 
    skladnik = x
    wynik = x
    for j in range(1,n+1):
        skladnik = skladnik * ((j)/(j + 1)) * x * (-1)
        wynik+=skladnik
    return wynik

#metoda Bablionska na znajdowanie sqrt
#przez rysowanie coraz bardziej "kwadratowych" prostokatow
def pierwiastek(x, eps = 0.0001):
    approx1 = 1
    approx2 = (approx1 + x)/2
    while abs(approx1 - approx2) > eps:
        approx1 = approx2
        approx2 = (approx1 + x/approx1)/2
    return approx2
"""

def pierwiastek2(x):
    if abs(x) > 1:
        odwrotnosc = True
        x = 1/x
    else:
        odwrotnosc = False
    if odwrotnosc:
        return power(e, 0.5*(-1)*ln(x+1))
"""
e = 2.71828182846

if __name__ == "__main__":
    print(ln(1/2*e))