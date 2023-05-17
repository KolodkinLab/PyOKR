# PyOKR
A Python-based optokinetic reflex analysis tool to measure and quantify eye tracking motion in three dimensions.
<img src="https://user-images.githubusercontent.com/131790844/234343487-2696a646-9977-4ce8-9b2f-6a7ee73a50ef.gif" width=45% height=45%>

**Requirements and Imports**: 

- Python 3.8

- PyQt5

- Spyder IDE (suggested for interactive graphs)

**Tutorial**:

<img src="https://user-images.githubusercontent.com/131790844/234955247-81bd3c4c-b7ef-422e-9e7f-9b093297557f.jpg" width = 75% height=75%>

**Panel A**: This displays the current CSV wave file path along with the current analysis output folder path. Wave files can be selected with Ctrl+O, the button in panel C, or the button under File. Output folders can be set with Ctrl+E, the button in panel C, or the button under File. Before starting analysis, set the mouse with Ctrl+S or the button under File. Finally, type the output file name into the text box (e.g. "GeneName_MouseNumber_Analysis"). Exporting the data in Panel G will save the final output file in the set Output folder under the entered file name.

**Panel B**: This panel allows setting the parameters of the recorded wave. Stimulus direction is either Horizontal or Vertical. The stimulus rotation is CW or CCW. This refers to the four possible stimuli directions: Horizontal CW = Forward, Horizontal CCW = Backward, Vertical CW = Downward, Vertical CCW = Forward. Stimulus speed in degrees/sec can be set with the text box.

**Panel C**: This panel has buttons to set the current CSV wave file and select the output folder. This is also available with Ctrl+O and Ctrl+E respectively.

**Panel D**: This button plots the wave form and automatically identifies saccades. Interactive plot allows for the addition and removal of individual dots via right-click and left-clicking. Middle-clicking sets the final points for eventual analysis.

**Panel E**: In this panel, top and bottoms of each saccade are automatically determined. Adjustments can be made through the same way as Panel D.

**Panel F**: In this panel, you can set the polynomial regression order as well as the distance between the estimated regression polynomial. A linear polynomial regression is recommended in calculation of final gains in relation to stimulus speed. The Final Analysis button will calculate gains based off of regression estimation. The 3D graph plots the graph in 3-dimensions, with time on the X-axis, horizontal eye motion on the Y-axis, and vertical eye motion on the Z-axis.

**Panel G**: In this panel, "Add Epoch" adds the analysis to the final dataset, if the final estimation looks accurate. "View Current Dataset" displays all epochs that have been added previously to the dataset for a mouse. "Export Data" saves the dataset and exports it to a finalized CSV based on parameters set in Panel A.

After all mouse data is collected, the button, "Sort Data," under Analysis will sort all final CSVs for a given experiment to allow for easier processing for generating a graph.
