#!/usr/bin/env python3
"""Module"""

while True:
    question = input("Q: ").lower()

    if question in ["exit", "quit", "goodbye", "bye"]:
        print("A: Goodbye")
        exit()
    else:
        print("A:")
