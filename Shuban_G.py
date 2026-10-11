class ShubanGolkonda:
    def __init__(self):
        self.SGbotLink1 = DHLink(d=0.135, a=0, alpha=0.0, qlim=[-pi*175/180, pi*175/180])
        self.SGbotLink2 = DHLink(d=0, a=0.0, alpha=-pi / 2, qlim=[-pi*70/180, pi*90/180], offset=pi/2)
        self.SGbotLink3 = DHLink(d=0, a=0.135, alpha=0.0, qlim=[-pi*135/180, pi*70/180])
        self.SGbotLink4 = DHLink(d=120, a=38, alpha=-pi / 2, qlim=[-pi*170/180, pi*170/180])
        self.SGbotLink5 = DHLink(d=0.0, a=0, alpha=pi / 2, qlim=[-pi*115/180, pi*115/180])
        self.SGbotLink6 = DHLink(d=0.070, a=0, alpha=-pi / 2, qlim=[-2 * pi, 2 * pi]),

        self.SGbot = DHRobot([self.SGbotLink1, self.SGbotLink2, self.SGbotLink3, self.SGbotLink4, self.SGbotLink5, self.SGbotLink6], name="SGbot")
        cyl_viz = CylindricalDHRobotPlot(self.Jbot, cylinder_radius=0.05, color="#3478f6")
        self.SGbot = cyl_viz.create_cylinders()

        self.env = sw.Swift()
        self.env.launch(realtime=True)
        self.stop_event = threading.Event()
