## Setup
Install the necessary software to run the program to control the camera. Since these cameras are old and no longer supported, we need to install two things. One of which provides the necessary drivers for me to be able to "see" the camera and the other one is a programming interface which allows me to control the camera with python rather than c++ (which makes it easier on me).

https://ids-imaging.org  go to software & documentation:  
- Install ids software suite == 4.97 (500M)  
- Install ids_peak SDK standard installation version 2.18.0

Download and install python  
- https://www.python.org/downloads/  
- I have version 3.13.5 installed but it might work with other versions

Download the source code here as a zip file and extract it to a location of your computer.

Download vscode:
- https://code.visualstudio.com/
- open it and install the python extension (just called "Python")
- 'File > Open folder...' and open the extracted source folder called "Worms"
- Ctrl+Shift+` to open a terminal window: use the following commands (should say PS next to the current folder)

```PS
    python --version #Should print 3.13 or greater
    python -m venv venv
    .\venv\Scripts\activate.bat
    pip install -r requirements,txt
```

- After you activate the envrionment your terminal line should have a `(venv)` next to it.  i.e. `(venv) PS D:\Repositories\Worms>`

Click the play button in the top left and the `Worms.py` script will run.
test