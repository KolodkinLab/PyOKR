# PyOKR
A Python-based optokinetic reflex analysis tool to measure and quantify eye tracking motion in three dimensions. Video-oculography data can be modeled computationally to quantify specific tracking speeds and ability in horizontal and vertical space.
<img src="https://user-images.githubusercontent.com/131790844/234343487-2696a646-9977-4ce8-9b2f-6a7ee73a50ef.gif" width=45% height=45%>

**Requirements**: 

- Python >= 3.8

- Spyder IDE (suggested for interactive graphs)

**Imports**:

- PyQT5

- Pandas

- Matplotlib

- Numpy

- Sklearn.neighbors (from scikit)

- Scipy

- SymPy

- Pandasgui

**Run**:
To run PyOKR, follow these steps:

1. Install PyOKR with "pip install PyOKR"
2. For Windows: run "from PyOKR import OKR_win as o". For Mac: run "from PyOKR import OKR_osx as o".
3. Run "o.run()" and the UI will appear
4. Running this in Spyder with popped out plots is necessary for graph supervision
