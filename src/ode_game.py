import requests
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, Toplevel
import os
from PIL import Image, ImageTk


def play_game(image_path, server_url, target_language='en'):
    """
    Plays the 'Guess the Object' game by interacting with the provided object detection server.

    Args:
        image_path (str): The full path to the image file.
        server_url (str): The URL of the object detection server.
        target_language (str): The language code for translation. Defaults to 'en'.

    Returns:
        None
    """
    # Step 1: Upload the image and get the detections
    try:
        with open(image_path, 'rb') as img_file:
            files = {'file': img_file}
            data = {
                'auto_select': 'true',
                'target_language': target_language
            }
            response = requests.post(f'{server_url}/get_detections', files=files, data=data)
            response.raise_for_status()
            detections = response.json()['detections']
    except Exception as e:
        messagebox.showerror("Error", f"Error occurred during detection: {str(e)}")
        return

    # Step 2: Start the game
    messagebox.showinfo("Game Start",
                        "Welcome to the 'Guess the Object' game! You have 3 attempts to guess the objects in the image.")

    hints = [f"The object is '{detection['name']}' in English." for detection in detections]

    correct_guesses = 0

    # Special effects label
    effect_label = tk.Label(root, font=('Helvetica', 16), fg='white', bg='black')
    effect_label.pack(pady=10)

    # Display the annotated image in a new window
    base_name = os.path.basename(image_path)
    annotated_image_path = os.path.join(os.path.dirname(image_path), f'annotated_{base_name}')
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

    # Step 3: Player guesses the objects
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

    # Step 4: End game and show results
    result_message = f"Game Over! You guessed {correct_guesses} out of {len(detections)} objects correctly."
    if correct_guesses == len(detections):
        result_message += "\nCongratulations! You guessed all the objects!"
    else:
        result_message += "\nBetter luck next time!"

    messagebox.showinfo("Game Over", result_message)


def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if file_path:
        play_game(file_path, server_url_entry.get(), language_var.get())  # Pass the full path directly


# Creating the GUI
root = tk.Tk()
root.title("Guess the Object Game")

# URL Entry
tk.Label(root, text="Server URL:").pack(pady=10)
server_url_entry = tk.Entry(root)
server_url_entry.pack(pady=5)
server_url_entry.insert(0, "http://localhost:5000")

# Language Selection
tk.Label(root, text="Select Language:").pack(pady=10)
language_var = tk.StringVar(value='en')
language_menu = tk.OptionMenu(root, language_var, 'en', 'es', 'fr', 'de', 'zh', 'ja',
                              'ko')  # Add supported languages here
language_menu.pack(pady=5)

# Upload Button
upload_button = tk.Button(root, text="Upload Image and Start Game", command=upload_image)
upload_button.pack(pady=20)

root.mainloop()
