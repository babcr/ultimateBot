"""Définition  d'un instrument financier.
"""

from ITick import QuoteTick
from collections import deque



class Instrument:
    def __init__(self, symbol, digits, tick_size):
        self.symbol = symbol
        self.digits = digits
        self.point = 10 ** -self.digits
        self.tick_size = tick_size
        self.historique_quotes = deque()  # Liste pour stocker l'historique des quotes du instrument

    def add_quote(self, quote: QuoteTick):
        self.historique_quotes.append(quote)

    def pop_quote(self):
        return self.historique_quotes.popleft() if self.historique_quotes else None