import requests
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, Toplevel
from tkinter import ttk
import os
from PIL import Image, ImageTk

def get_supported_languages(server_url):
    """
    Retrieves the list of supported languages from the server.

    Args:
        server_url (str): The URL of the object detection server.

    Returns:
        list: A list of supported language codes (e.g., ['en', 'es', 'fr', ...]).
    """
    try:
        response = requests.get(f'{server_url}/supported_languages')
        response.raise_for_status()
        languages = response.json().get('supported_languages', [])
        return languages
    except Exception as e:
        messagebox.showerror("Error", f"Error occurred while fetching supported languages: {str(e)}")
        return []


def play_game(image_path, server_url, source_language, target_language):
    """
    Plays the 'Guess the Object' game by interacting with the provided object detection server.

    Args:
        image_path (str): The full path to the image file.
        server_url (str): The URL of the object detection server.
        source_language (str): The language code to translate from.
        target_language (str): The language code to translate to.

    Returns:
        None
    """
    # Step 1: Upload the image and get the annotated image
    try:
        with open(image_path, 'rb') as img_file:
            files = {'file': img_file}
            data = {
                'auto_select': 'true',
                'source_language': source_language,
                'target_language': target_language
            }
            response = requests.post(f'{server_url}/detect', files=files, data=data)
            response.raise_for_status()

            # Save the annotated image locally
            annotated_image_path = os.path.join(os.path.dirname(image_path), f'annotated_{os.path.basename(image_path)}')
            with open(annotated_image_path, 'wb') as f:
                f.write(response.content)
    except Exception as e:
        messagebox.showerror("Error", f"Error occurred during image detection: {str(e)}")
        return

    # Step 2: Get detections separately
    try:
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
                messagebox.showerror("Error", "No detections found in the response.")
                return
    except Exception as e:
        messagebox.showerror("Error", f"Error occurred while fetching detections: {str(e)}")
        return

    # Step 3: Start the game
    messagebox.showinfo("Game Start", "Welcome to the 'Guess the Object' game! You have 3 attempts to guess the objects in the image.")

    hints = [f"The object is '{detection['name']}' in English." for detection in detections]

    correct_guesses = 0

    # Special effects label
    effect_label = tk.Label(root, font=('Helvetica', 16), fg='white', bg='black')
    effect_label.pack(pady=10)

    # Display the annotated image in a new window
    try:
        annotated_img = Image.open(annotated_image_path)

        # Set maximum dimensions for the window
        max_width, max_height = 800, 600  # Define the maximum width and height for the image window

        # Scale the image if it exceeds the maximum dimensions
        img_width, img_height = annotated_img.size
        if img_width > max_width or img_height > max_height:
            scale_factor = min(max_width / img_width, max_height / img_height)
            img_width = int(img_width * scale_factor)
            img_height = int(img_height * scale_factor)
            annotated_img = annotated_img.resize((img_width, img_height))

        # Create a new window for displaying the annotated image
        img_window = Toplevel(root)
        img_window.title("Annotated Image")
        img_window.geometry(f"{img_width}x{img_height}")

        # Display the image in the new window
        img_tk = ImageTk.PhotoImage(annotated_img)
        img_label = tk.Label(img_window, image=img_tk)
        img_label.image = img_tk  # Keep a reference to avoid garbage collection
        img_label.pack()
    except Exception as e:
        messagebox.showwarning("Warning", f"Failed to display annotated image: {str(e)}")

    # Step 4: Player guesses the objects
    for i, detection in enumerate(detections):
        revealed_name = detection['translated_name'][0]  # Start with the first letter
        for attempt in range(3):
            # Update the hint to show the revealed part of the translated name
            current_hint = f"Hint {i + 1}: The object is '{detection['name']}' in English. (Revealed: {revealed_name})"
            hints[i] = current_hint
            messagebox.showinfo("Hints", f"Here are your hints:\n\n" + "\n".join(hints))

            # Ask the user to guess
            question = f"Attempt {attempt + 1}: What is object '{detection['name']}' in {target_language.upper()}?"
            guess = simpledialog.askstring("Guess the Object", question)
            if not guess:
                messagebox.showwarning("Invalid Input", "Please enter a valid guess.")
                continue

            if guess.strip().lower() == detection['translated_name'].lower():
                effect_label.config(text="Correct!", fg='green')
                root.update()
                root.after(1000, effect_label.config, {'text': '', 'fg': 'white'})  # Clear the effect after 1 second
                correct_guesses += 1
                break
            else:
                effect_label.config(text="Incorrect!", fg='red')
                root.update()
                root.after(1000, effect_label.config, {'text': '', 'fg': 'white'})  # Clear the effect after 1 second
                if len(revealed_name) < len(detection['translated_name']):
                    revealed_name += detection['translated_name'][len(revealed_name)]  # Reveal one more letter
                messagebox.showinfo("Incorrect", "Incorrect. Try again.")

        if correct_guesses == len(detections):
            break

    # Step 5: End game and show results
    result_message = f"Game Over! You guessed {correct_guesses} out of {len(detections)} objects correctly."
    if correct_guesses == len(detections):
        result_message += "\nCongratulations! You guessed all the objects!"
    else:
        result_message += "\nBetter luck next time!"

    messagebox.showinfo("Game Over", result_message)


def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if file_path:
        play_game(file_path, server_url_entry.get(), source_language_var.get(), target_language_var.get())


# Creating the GUI
root = tk.Tk()
root.title("Guess the Object Game")

# Create a canvas for scrolling
canvas = tk.Canvas(root)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Create a vertical scrollbar linked to the canvas
scrollbar = tk.Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Create a frame inside the canvas which will hold the widgets
frame = tk.Frame(canvas)
canvas.create_window((0, 0), window=frame, anchor='nw')

# Update the scrollbar and frame size when the window is resized
def on_frame_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

frame.bind("<Configure>", on_frame_configure)

# URL Entry
tk.Label(frame, text="Server URL:").pack(pady=10)
server_url_entry = tk.Entry(frame)
server_url_entry.pack(pady=5)
server_url_entry.insert(0, "http://localhost:5000")

# Fetch supported languages dynamically
supported_languages = get_supported_languages(server_url_entry.get())

# Source Language Selection
tk.Label(frame, text="Translate From (Source Language):").pack(pady=10)
source_language_var = tk.StringVar(value='en')
source_language_menu = ttk.Combobox(frame, textvariable=source_language_var, values=supported_languages)
source_language_menu.pack(pady=5)
source_language_menu.set('en')  # Set default value

# Target Language Selection
tk.Label(frame, text="Translate To (Target Language):").pack(pady=10)
target_language_var = tk.StringVar(value='en')
target_language_menu = ttk.Combobox(frame, textvariable=target_language_var, values=supported_languages)
target_language_menu.pack(pady=5)
target_language_menu.set('en')  # Set default value

# Upload Button
upload_button = tk.Button(frame, text="Upload Image and Start Game", command=upload_image)
upload_button.pack(pady=20)

# Configure scrollbar
canvas.configure(yscrollcommand=scrollbar.set)
scrollbar.config(command=canvas.yview)

root.mainloop()
