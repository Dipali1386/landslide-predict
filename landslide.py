import os

# 📁 Path to your image folder
folder_path = "C:/Users/dipali/OneDrive/Desktop/images"

# ✅ Get all files with valid image extensions
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# ✅ Sort for consistent renaming order
image_files.sort()


# ✅ Check how many files are found
print(f"Found {len(image_files)} image files to rename.\n")

# ✅ Rename loop
for i, filename in enumerate(image_files, start=1):
    ext = os.path.splitext(filename)[1].lower()          # Get the original extension
    new_name = f"{i:03}{ext}"                            # Format: 001.jpg, 002.jpg, ...
    old_path = os.path.join(folder_path, filename)
    new_path = os.path.join(folder_path, new_name)

    # Skip if already correctly named
    if filename == new_name:
        print(f"⏩ Already correct: {filename}")
        continue

    # Rename and confirm
    try:
        os.rename(old_path, new_path)
        print(f"✅ Renamed: {filename} → {new_name}")
    except Exception as e:
        print(f"❌ Error renaming {filename}: {e}")

print("\n🎉 All image files renamed safely!")