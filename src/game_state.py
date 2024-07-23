# game_state.py

class HangmanGame:
    def __init__(self, word):
        self.word = word
        self.correct_guesses = set()
        self.incorrect_guesses = set()
        self.max_incorrect_guesses = 6  # You can set the number of allowed incorrect guesses

    def guess(self, letter):
        letter = letter.lower()
        if letter in self.word:
            self.correct_guesses.add(letter)
        else:
            self.incorrect_guesses.add(letter)

    def get_display_word(self):
        return ''.join([letter if letter in self.correct_guesses else '_' for letter in self.word])

    def is_game_over(self):
        return self.is_won() or self.is_lost()

    def is_won(self):
        return all(letter in self.correct_guesses for letter in self.word)

    def is_lost(self):
        return len(self.incorrect_guesses) >= self.max_incorrect_guesses

    def get_incorrect_guesses(self):
        return list(self.incorrect_guesses)

    def get_correct_guesses(self):
        return list(self.correct_guesses)
