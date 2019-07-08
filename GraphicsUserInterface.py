
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


    def draw(self, imageReal : list, imageObserved : list, width1 : int, height1 : int, width2 : int, height2 : int, 
                         semantics : dict, texts : list, lambdaNewCaps, lambdaTrainCaps, lambdaNewAttr, lambdaTrainAttr, 
                         save : bool = False, recommendation : str = None):
        # semantics    # Observation - List of Semantics

        selectedObs = []
        newName = [""]  # Faking Pointers..

        def mouseClick(event):
            if event.xdata is None or event.ydata is None or event.button is None:
                # No Interesting Data..
                return
            if event.inaxes != axarr[0][2]:
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

        def runButtonA(event):        
            if len(selectedObs) > 0 and len(newName[0]) > 0:
                lambdaNewCaps(newName[0], selectedObs)
                plt.close()

        def runButtonB(event):        
            if len(selectedObs) > 0 and len(newName[0]) > 0:
                lambdaTrainCaps(newName[0], selectedObs)
                plt.close()

        def runButtonC(event):        
            if len(selectedObs) > 0 and len(newName[0]) > 0:
                lambdaNewAttr(newName[0], selectedObs)
                plt.close()

        def runButtonD(event):        
            if len(selectedObs) > 0 and len(newName[0]) > 0:
                lambdaTrainAttr(newName[0], selectedObs)
                plt.close()

        def onTextSubmit(text):
            newName[0] = text

        pixels1 = [0.0] * (width1 * height1 * 3)
        pixels2 = [0.0] * (width2 * height2 * 3)
        
        for yy in range(height1):
            for xx in range(width1):
                pixels1[(yy * width1 + xx) * 3] = imageReal[(yy * width1 + xx) * 4]
                pixels1[(yy * width1 + xx) * 3 + 1] = imageReal[(yy * width1 + xx) * 4]
                pixels1[(yy * width1 + xx) * 3 + 2] = imageReal[(yy * width1 + xx) * 4]

        for yy in range(height2):
            for xx in range(width2):
                pixels2[(yy * width2 + xx) * 3] = imageObserved[(yy * width2 + xx) * 4]
                pixels2[(yy * width2 + xx) * 3 + 1] = imageObserved[(yy * width2 + xx) * 4]
                pixels2[(yy * width2 + xx) * 3 + 2] = imageObserved[(yy * width2 + xx) * 4]


        fig, axarr = plt.subplots(2,3)
        imageData = numpy.reshape(pixels1, [height1, width1, 3])
        axarr[0][0].imshow(imageData)
        axarr[0][1].imshow(numpy.reshape(pixels2, [height2, width2, 3]))
        axarr[0][2].imshow(numpy.reshape(pixels2, [height2, width2, 3]))
        axarr[0][0].set_axis_off()
        axarr[0][1].set_axis_off()
        axarr[0][2].set_axis_off()
        axarr[0][0].set_title("Original")
        axarr[0][1].set_title("Internal Represenation")
        axarr[0][2].set_title("Semantics")

        # Hide lower Row to make room for Meta-learning
        axarr[1][0].set_axis_off()
        axarr[1][1].set_axis_off()
        axarr[1][2].set_axis_off()

        for semanticList in semantics.values():
            for semantic in semanticList:
                axarr[0][2].add_patch(semantic)

        for text in texts:
            axarr[0][2].text(text[0], text[1], text[2], color = 'y', fontsize=8)

        if save is True:
            scipy.misc.imsave("scene.png", imageData)


        if recommendation is not None:
            # Meta-Learning

            fig.canvas.mpl_connect('button_press_event', mouseClick)

            axdesc = plt.axes([0.03, 0.475, 0.94, 0.1])
            axdesc.set_axis_off()
            axdesc.text(0, 0.0, "(Select or Deselect (LMB) Primitives in 'Semantics' Plot to be combined into a new or existing Semantic Capsule \n and then choose one of the four options below, optionally following the recommendation by the Meta-learning agent)", fontsize=7, wrap=True)

            axrec = plt.axes([0.15, 0.4, 0.8, 0.1])
            axrec.set_axis_off()
            axrec.text(0, 0.0, "Recommendation: " + recommendation, fontsize=10, wrap=True, bbox=dict(facecolor='red', alpha=0.2))


            # New Capsule
            axboxA = plt.axes([0.3, 0.25, 0.35, 0.075])
            textBoxA = TextBox(axboxA, 'New Capsule Name', initial='')
            textBoxA.on_submit(onTextSubmit)

            axbtnA = plt.axes([0.65, 0.25, 0.25, 0.075])
            bnextA = Button(axbtnA, 'Train New Capsule')
            bnextA.on_clicked(runButtonA)
            

            # Existing Capsule
            axboxB = plt.axes([0.3, 0.175, 0.35, 0.075])
            textBoxB = TextBox(axboxB, 'Existing Capsule Name', initial='')
            textBoxB.on_submit(onTextSubmit)

            axbtnB = plt.axes([0.65, 0.175, 0.25, 0.075])
            bnextB = Button(axbtnB, 'Train Exist. Caps.')
            bnextB.on_clicked(runButtonB)
            

            # New Attribute
            axboxC = plt.axes([0.3, 0.1, 0.35, 0.075])
            textBoxC = TextBox(axboxC, 'New Attribute Name', initial='')
            textBoxC.on_submit(onTextSubmit)

            axbtnC = plt.axes([0.65, 0.1, 0.25, 0.075])
            bnextC = Button(axbtnC, 'Train new Attribute')
            bnextC.on_clicked(runButtonC)
            

            # Existing Attribute
            axboxD = plt.axes([0.3, 0.025, 0.35, 0.075])
            textBoxD = TextBox(axboxD, 'Existing Attribute Name', initial='')
            textBoxD.on_submit(onTextSubmit)

            axbtnD = plt.axes([0.65, 0.025, 0.25, 0.075])
            bnextD = Button(axbtnD, 'Train Exist. Attr.')
            bnextD.on_clicked(runButtonD)
            
        plt.show()
    
        

    def drawMovie(self, frames : list, width : int, height : int, deltaT : float, save : bool):
        # frames        # List of List of Pixels

        fig = plt.figure()

        images = []
        for idx, frame in enumerate(frames):
            pixels = [0.0] * (width * height * 3)
            
            for yy in range(height):
                for xx in range(width):
                    pixels[(yy * width + xx) * 3] = frame[(yy * width + xx) * 4]
                    pixels[(yy * width + xx) * 3 + 1] = frame[(yy * width + xx) * 4]
                    pixels[(yy * width + xx) * 3 + 2] = frame[(yy * width + xx) * 4]

            imageData = numpy.reshape(pixels, [height, width, 3])
            newImage = plt.imshow(imageData)
            images.append([newImage])

            if save is True:
                scipy.misc.imsave("videoframe" + str(idx) + ".png", imageData)

        fullAnim = animation.ArtistAnimation(fig, images, interval=deltaT * 1000, repeat_delay=0,
                                        blit=True)
                      
        plt.show()
    