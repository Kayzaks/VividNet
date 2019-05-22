
from matplotlib.widgets import Button
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.widgets import TextBox
import numpy
import scipy.misc

class GraphicsUserInterface:
        
    def identifyObservation(self, semantics : dict, xpos : float, ypos : float):
        # semantics    # Observation - List of Semantics
        totList = []
        for observation, semanticList in semantics.items():
            for semantic in semanticList:
                if type(semantic) == patches.Rectangle:
                    if xpos >= semantic.get_x() and xpos <= semantic.get_x() + semantic.get_width():
                        if ypos >= semantic.get_y() and ypos <= semantic.get_y() + semantic.get_height():
                            totList.append((observation, semantic)) 
        return totList # List of (Observation , patches.Rectangle)


    def loadImage(self, filename : str):
        image = scipy.misc.imread(filename)

        width = len(image)
        height = len(image[0])

        outImage = [0.0] * width * height * 4

        for yy in range(height):
            for xx in range(width):
                outImage[(yy * width + xx) * 4] = float(image[yy][xx][0]) / 255.0
                outImage[(yy * width + xx) * 4 + 1] = float(image[yy][xx][1]) / 255.0
                outImage[(yy * width + xx) * 4 + 2] = float(image[yy][xx][2]) / 255.0
                outImage[(yy * width + xx) * 4 + 3] = 0.0

        return outImage


    def draw(self, pixels1 : list, pixels2 : list, width1 : int, height1 : int, width2 : int, height2 : int, 
                         semantics : dict, texts : list, lambdaTrain, save : bool = False):
        # semantics    # Observation - List of Semantics

        selectedObs = []
        newName = [""]  # Faking Pointers..

        def mouseClick(event):
            if event.xdata is None or event.ydata is None or event.button is None:
                # No Interesting Data..
                return
            if event.inaxes != axarr[2]:
                # Wrong Axis
                return

            obsSemList = self.identifyObservation(semantics, event.xdata, event.ydata)

            if obsSemList is False:
                return

            for obs, semantic in obsSemList:
                if event.button == 1 and obs not in selectedObs:
                    # Left Mouse Button -> Add
                    selectedObs.append(obs)
                    semantic.set_edgecolor('blue')
                elif event.button == 1 and obs in selectedObs:
                    # Right Mouse Button -> Remove
                    selectedObs.remove(obs)
                    semantic.set_edgecolor('yellow')

            fig.canvas.draw()

        def runButton(event):        
            if len(selectedObs) > 0 and len(newName[0]) > 0:
                lambdaTrain(newName[0], selectedObs)
                plt.close()


        def onTextSubmit(text):
            newName[0] = text

        fig, axarr = plt.subplots(1,3)
        imageData = numpy.reshape(pixels1, [height1, width1, 3])
        axarr[0].imshow(imageData)
        axarr[1].imshow(numpy.reshape(pixels2, [height2, width2, 3]))
        axarr[2].imshow(numpy.reshape(pixels2, [height2, width2, 3]))
        axarr[0].set_axis_off()
        axarr[1].set_axis_off()
        axarr[2].set_axis_off()
        axarr[0].set_title("Original")
        axarr[1].set_title("Internal Represenation")
        axarr[2].set_title("Semantics")

        for semanticList in semantics.values():
            for semantic in semanticList:
                axarr[2].add_patch(semantic)

        for text in texts:
            axarr[2].text(text[0], text[1], text[2], color = 'y', fontsize=8)

        if save is True:
            scipy.misc.imsave("scene.png", imageData)

        fig.canvas.mpl_connect('button_press_event', mouseClick)
        axbtn = plt.axes([0.81, 0.05, 0.1, 0.075])
        bnext = Button(axbtn, 'Train')
        bnext.on_clicked(runButton)

        
        axbox = plt.axes([0.6, 0.15, 0.35, 0.075])
        textBox = TextBox(axbox, 'Name', initial='')
        textBox.on_submit(onTextSubmit)
        
        axdesc = plt.axes([0.1, 0.05, 0.55, 0.1])
        axdesc.set_axis_off()
        axdesc.text(0, 0.0, "Select or Deselect (LMB) Primitives in 'Semantics' Plot \n to be combined into a new Semantic Capsule, enter \n a name and press 'Train'", fontsize=10, wrap=True)

        plt.show()
    
        

    def drawMovie(self, frames : list, width : int, height : int, deltaT : float, save : bool):
        # frames        # List of List of Pixels

        fig = plt.figure()

        images = []
        for idx, frame in enumerate(frames):
            imageData = numpy.reshape(frame, [height, width, 3])
            newImage = plt.imshow(numpy.reshape(frame, [height, width, 3]))
            images.append([newImage])

            if save is True:
                scipy.misc.imsave("videoframe" + str(idx) + ".png", imageData)

        fullAnim = animation.ArtistAnimation(fig, images, interval=deltaT * 1000, repeat_delay=0,
                                        blit=True)
                      
        plt.show()
    