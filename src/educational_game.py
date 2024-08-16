import requests
import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel
from tkinter import ttk
import os
from PIL import Image, ImageTk
import random

# Global variable to store the PhotoImage object
img_tk_global = None

from graphviz import Digraph

# Create a new directed graph with the 'dot' layout engine for orthogonal layout
dot = Digraph(format='png', engine='dot')

# Set global graph attributes for a clearer tree layout
dot.attr(dpi='100', rankdir='TB', style='solid')

# Add nodes with descriptions
dot.node('Start', 'Start')
dot.node('Upload', 'Upload Image')
dot.node('SelectLang', 'Select Source and Target Language')
dot.node('FetchLang', 'Fetch Supported Languages')
dot.node('Loading', 'Show Loading Screen')
dot.node('SendDetect', 'Send Image to Backend for Detection')
dot.node('ReceiveImage', 'Receive Annotated Image')
dot.node('CheckError', 'Check for Detection Errors')
dot.node('DisplayImage', 'Display Annotated Image')
dot.node('InitGame', 'Initialize Game')
dot.node('Guess', 'User Makes Guess')
dot.node('CheckGuess', 'Check Guess Against Correct Answer')
dot.node('Correct', 'Correct Guess')
dot.node('Incorrect', 'Incorrect Guess')
dot.node('UpdateScore', 'Update Score')
dot.node('Decrement', 'Decrement Attempts')
dot.node('ProvideHint', 'Provide Hint (if attempts left)')
dot.node('AllProcessed', 'All Detections Processed?')
dot.node('ShowResult', 'Show Game Result')
dot.node('Repeat', 'Repeat Guessing for Remaining Detections')

# Add edges to connect nodes
dot.edge('Start', 'Upload')
dot.edge('Upload', 'SelectLang')
dot.edge('SelectLang', 'FetchLang')
dot.edge('FetchLang', 'Loading')
dot.edge('Loading', 'SendDetect')
dot.edge('SendDetect', 'ReceiveImage')
dot.edge('ReceiveImage', 'CheckError')
dot.edge('CheckError', 'DisplayImage')
dot.edge('DisplayImage', 'InitGame')
dot.edge('InitGame', 'Guess')
dot.edge('Guess', 'CheckGuess')
dot.edge('CheckGuess', 'Correct', label='Correct Guess')
dot.edge('CheckGuess', 'Incorrect', label='Incorrect Guess')
dot.edge('Correct', 'UpdateScore')
dot.edge('Incorrect', 'Decrement')
dot.edge('Decrement', 'ProvideHint')
dot.edge('ProvideHint', 'Guess', label='Attempts Left')
dot.edge('UpdateScore', 'AllProcessed')
dot.edge('AllProcessed', 'ShowResult', label='Yes')
dot.edge('AllProcessed', 'Repeat', label='No')
dot.edge('Repeat', 'Guess')

# Render the decision tree
dot.render('ortho_game_procedures', format='png', cleanup=True)

def get_supported_languages(server_url):
    try:
        response = requests.get(f'{server_url}/supported_languages')
        response.raise_for_status()
        languages = response.json().get('supported_languages', [])
        return languages
    except requests.RequestException as e:
        messagebox.showerror("Error", f"Error occurred while fetching supported languages: {str(e)}")
        return []

def show_loading_screen():
    loading_window = Toplevel(root)
    loading_window.title("Loading")
    loading_window.geometry("300x100")
    loading_window.grab_set()  # Make this window modal

    label = tk.Label(loading_window, text="Loading... Please wait.", font=("Helvetica", 16))
    label.pack(expand=True)

    # Add a close button to the loading screen
    close_button = tk.Button(loading_window, text="Close", command=loading_window.destroy)
    close_button.pack(pady=10)

    return loading_window

def show_disclaimer_and_credits():
    info_window = Toplevel(root)
    info_window.title("Disclaimer and Credits")
    info_window.geometry("600x400")

    disclaimer_text = (
        "This application uses the YOLOv10 model for object detection. Please do not upload any sensitive or personal images. "
        "The application is not intended for commercial use. The maximum upload size is 16 MB."
    )

    credits_text = (
        "Credits:\n"
        "- Faycal Kilali\n"
        "- YOLOv10: Real-Time End-to-End Object Detection by Ao Wang, Hui Chen, Lihao Liu, Kai Chen, Zijia Lin, Jungong Han, Guiguang Ding. "
        "ArXiv preprint arXiv:2405.14458, 2024"
    )

    text = f"{disclaimer_text}\n\n{credits_text}"

    # Create a Text widget for better formatting
    text_widget = tk.Text(info_window, wrap=tk.WORD, padx=10, pady=10, font=("Helvetica", 12))
    text_widget.insert(tk.END, text)
    text_widget.config(state=tk.DISABLED)  # Make the text widget read-only
    text_widget.pack(expand=True, fill=tk.BOTH)

def play_game(image_path, server_url, source_language, target_language):
    global img_tk_global  # Declare it as global to persist

    # Show loading screen
    loading_window = show_loading_screen()

    try:
        # Detect and download the annotated image
        with open(image_path, 'rb') as img_file:
            files = {'file': img_file}
            data = {
                'auto_select': 'true',
                'source_language': source_language,
                'target_language': target_language
            }
            response = requests.post(f'{server_url}/detect', files=files, data=data)
            response.raise_for_status()
            annotated_image_path = os.path.join(os.path.dirname(image_path),
                                                f'annotated_{os.path.basename(image_path)}')
            with open(annotated_image_path, 'wb') as f:
                f.write(response.content)
    except requests.RequestException as e:
        loading_window.destroy()
        messagebox.showerror("Error", f"Error occurred during image detection: {str(e)}")
        return

    try:
        # Get the detections from the server
        with open(image_path, 'rb') as img_file:
            files = {'file': img_file}
            data = {
                'auto_select': 'true',
                'source_language': source_language,
                'target_language': target_language
            }
            detection_response = requests.post(f'{server_url}/get_detections', files=files, data=data)
            detection_response.raise_for_status()
            detections = detection_response.json().get('detections', [])
            if not detections:
                loading_window.destroy()
                messagebox.showerror("Error", "No detections found in the response.")
                return
    except requests.RequestException as e:
        loading_window.destroy()
        messagebox.showerror("Error", f"Error occurred while fetching detections: {str(e)}")
        return

    # Close loading screen and show game
    loading_window.destroy()
    messagebox.showinfo("Game Start",
                        "Welcome to the 'Guess the Object' game! You have up to 3 attempts to guess each object.")

    correct_guesses = 0
    max_attempts = 3

    try:
        # Open and resize the annotated image
        annotated_img = Image.open(annotated_image_path)
        max_width, max_height = 800, 600
        img_width, img_height = annotated_img.size
        if img_width > max_width or img_height > max_height:
            scale_factor = min(max_width / img_width, max_height / img_height)
            img_width = int(img_width * scale_factor)
            img_height = int(img_height * scale_factor)
            annotated_img = annotated_img.resize((img_width, img_height))

        # Create the annotated image window
        img_window = Toplevel(root)
        img_window.title("Annotated Image")
        img_window.geometry(f"{img_width + 50}x{img_height + 300}")  # Adjust dimensions to accommodate scroll

        img_frame = tk.Frame(img_window)
        img_frame.pack(fill=tk.BOTH, expand=True)

        img_canvas = tk.Canvas(img_frame, bg='white')
        img_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(img_frame, orient=tk.VERTICAL, command=img_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        img_canvas.configure(yscrollcommand=scrollbar.set)

        img_tk_global = ImageTk.PhotoImage(annotated_img)  # Store it in a global variable
        img_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk_global)

        # Bind the canvas to the frame to handle scrolling
        def on_canvas_configure(event):
            img_canvas.configure(scrollregion=img_canvas.bbox("all"))

        img_canvas.bind("<Configure>", on_canvas_configure)

        entry_vars = []
        attempt_vars = []

        # Initialize guesses and attempts
        user_guesses = [set() for _ in detections]
        attempts_left = [max_attempts] * len(detections)
        correct_answers = [detection['translated_name'].lower() for detection in detections]
        revealed_letters = [set() for _ in detections]  # Track revealed letters

        # Function to reveal a letter as a hint
        def reveal_hint(correct_answer, guessed_letters, revealed_letters):
            unrevealed_indices = [i for i, c in enumerate(correct_answer) if c not in guessed_letters and c != ' ']
            if unrevealed_indices:
                index = random.choice(unrevealed_indices)
                revealed_letters.add(index)
                return index, correct_answer[index]
            return None, None

        def update_word_display(word, guessed_letters, revealed_letters):
            return ' '.join(c if c in guessed_letters or i in revealed_letters else '_' for i, c in enumerate(word))

        def check_guesses():
            nonlocal correct_guesses
            for i, detection in enumerate(detections):
                user_guess = entry_vars[i].get().strip().lower()
                if user_guess == correct_answers[i]:
                    entry_vars[i].set("")
                    attempt_vars[i].config(text=f"Correct! The {detection['name']} object.", bg='lightgreen')
                    correct_guesses += 1
                else:
                    attempts_left[i] -= 1
                    if attempts_left[i] <= 0:
                        attempt_vars[i].config(text=f"Correct Answer: {correct_answers[i]}", bg='lightcoral')
                    else:
                        user_guesses[i].add(user_guess)
                        hint_index, hint_letter = reveal_hint(correct_answers[i], user_guesses[i], revealed_letters[i])
                        hint_display = update_word_display(correct_answers[i], user_guesses[i], revealed_letters[i])
                        if hint_letter:
                            hint_display = hint_display[:hint_index * 2] + hint_letter + hint_display[(hint_index * 2) + 1:]
                        attempt_vars[i].config(text=f"Attempts left: {attempts_left[i]} - Hint: {hint_display}",
                                               bg='lightyellow')

            if correct_guesses == len(detections):
                messagebox.showinfo("Congratulations!", "You guessed all the objects correctly!")
            elif all(attempts_left[i] <= 0 for i in range(len(detections))):
                messagebox.showinfo("Game Over", f"You guessed {correct_guesses} out of {len(detections)} correctly.")

        # Submit button
        submit_button = tk.Button(img_window, text=f"Submit {target_language} translations", command=check_guesses,
                                  bg='lightblue')
        submit_button.pack(pady=10)

        # Placing entry boxes and hint labels below the submit button
        for i, detection in enumerate(detections):
            entry_var = tk.StringVar()
            entry_vars.append(entry_var)
            label = tk.Label(img_window, text=f"Translate the {detection['name']} object:", bg='lightgoldenrodyellow')
            label.pack(pady=5, anchor='center')

            attempt_var = tk.Label(img_window, text="", bg='lightyellow')
            attempt_var.pack(pady=5, anchor='center')
            attempt_vars.append(attempt_var)

            entry = tk.Entry(img_window, textvariable=entry_var, width=50, justify='center')
            entry.pack(pady=5)

            # Initialize hint display
            hint_display = update_word_display(correct_answers[i], user_guesses[i], revealed_letters[i])
            attempt_var.config(text=f"Attempts left: {max_attempts} - Hint: {hint_display}")

    except Exception as e:
        messagebox.showwarning("Warning", f"Failed to display annotated image: {str(e)}")

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if file_path:
        play_game(file_path, server_url_entry.get(), source_language_var.get(), target_language_var.get())

root = tk.Tk()
root.title("AI Object Detection Framework and Educational Tool - Faycal Kilali")
root.geometry("600x600")

canvas = tk.Canvas(root, bg='lightblue')
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar = tk.Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

frame = tk.Frame(canvas, bg='lightblue')
canvas.create_window((0, 0), window=frame, anchor='nw')

def on_frame_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

frame.bind("<Configure>", on_frame_configure)

tk.Label(frame, text="Server URL:", bg='lightblue').pack(pady=10)
server_url_entry = tk.Entry(frame)
server_url_entry.pack(pady=5)
server_url_entry.insert(0, "http://localhost:5000")

# Initialize supported languages
supported_languages = get_supported_languages(server_url_entry.get())

tk.Label(frame, text="Translate From (Source Language):", bg='lightblue').pack(pady=10)
source_language_var = tk.StringVar(value='en')
source_language_menu = ttk.Combobox(frame, textvariable=source_language_var, values=supported_languages)
source_language_menu.pack(pady=5)
source_language_menu.set('en')

tk.Label(frame, text="Translate To (Target Language):", bg='lightblue').pack(pady=10)
target_language_var = tk.StringVar(value='en')
target_language_menu = ttk.Combobox(frame, textvariable=target_language_var, values=supported_languages)
target_language_menu.pack(pady=5)
target_language_menu.set('en')

upload_button = tk.Button(frame, text="Upload Image and Start Game", command=upload_image, bg='lightgreen')
upload_button.pack(pady=20)

info_button = tk.Button(frame, text="Disclaimer and Credits", command=show_disclaimer_and_credits, bg='lightyellow')
info_button.pack(pady=10)

canvas.configure(yscrollcommand=scrollbar.set)
scrollbar.config(command=canvas.yview)

root.mainloop()
