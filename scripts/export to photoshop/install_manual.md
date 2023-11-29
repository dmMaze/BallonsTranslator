# Export to photoshop script

## Installation

There are 2 ways to install.

1. Simply open the jsx file via `File -> Scripts -> Open... ` and select the script manually. The downside is that doing this every time is stupid and not convenient.
2. Place the jsx file in `Disk:\Program Files\Adobe\Adobe Photoshop [Version]]\Presets\Scripts`. The script will be displayed in the `File -> Scripts` interface
3. Auto installer (coming soon). Since the script will be updated, I’ll add a small script a little later that will put everything inside Photoshop itself. In theory, I’ll even make it as a separate plugin, if I don’t break my computer from not understanding the Adobe documentation

## Usage

1. Run the script (more details in the picture)</br>
![1700864913586](https://github.com/bropines/BallonsTranslator/assets/57861007/94bbc2de-24da-41f8-8f4c-94982d57e987)
2. Select your project's JSON file.
3. From the proposed list, select your image (which is open in PS) (more details in the picture)
![1700865117911](https://github.com/bropines/BallonsTranslator/assets/57861007/d9123072-72f0-48cf-84bf-b19a234bdf8b)
4. In the window that opens, import options, select the desired settings. An explanation of the settings will be below.
5. Done, he will think a little and arrange all the blocks almost like in BT

### Explanation of settings

**Import original and translation** - import the original and translation blocks, respectively. Import works either separately or together; 2 versions of blocks are created.

**Hide original and hide translation** - if you need the blocks to be hidden after import (for example, if you selected both import options, then these checkboxes will hide the visibility of the blocks in the layers panel)

**Use block text** - use block closed and block open (yes, I'm a little stupid with names xD). How are they different?
Just look at the GIF:
![1700865117922](https://github.com/bropines/BallonsTranslator/assets/57861007/8a2d639a-181d-4292-80ec-8a37bf778006)


## Bugs

I will warn you that I am just learning JS and my skills are not omnipotent, plus I don’t have much time. Well, Adobe also has extremely unclear documentation -_-.

- [ ] The font is not imported. I roughly understand why, but I don’t understand how to fix it.
- [ ] Text effects such as Italic, Bold and Underline are not imported.
- [ ] Due to the lack of “All caps” functions in BT and, in principle, most of the character settings, they are not available in import. Maybe I’ll add a checkbox or normal buttons that do this at the stage of preparing for import.
- [ ] Speed. Since in fact I am feeding bare JSON, the script needs time to read and extract data. Maybe we can speed this up...

#### Made by the crooked hands of @bropines 
