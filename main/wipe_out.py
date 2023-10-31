import os

def delete_contents(folder_path):
    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Wipe these directories out!
    directories = [
        'recognizer',
        'dataset/Original',
        'dataset/Resized',
        'dataset/Resizedandhist',
        'dataset/Hist',
        'dataset/Edge',
        'dataset/ResizedandSmall',
        'dataset/ResizedSmallHist',
        'dataset'
    ]

    # Function to delete folder contents
    def delete_contents(path):
        if os.path.exists(path) and os.path.isdir(path):
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    os.rmdir(item_path)
        else:
            print(f"The specified folder '{path}' does not exist or is not a valid directory.")

    # Loop through the directories and delete their contents
    for directory in directories:
        path = os.path.join(script_dir, directory)
        delete_contents(path)
