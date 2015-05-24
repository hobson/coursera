def create_video_pipeline(self):
    """Set up the video pipeline and the communication bus bewteen the video stream and gtk DrawingArea """
    video_pipeline = 'v4l2src device=/dev/video1 ! video/x-raw-yuv,width=640,height=480,framerate=30/1 ! xvimagesink'
    self.video_player = gst.parse_launch(video_pipeline) # create pipeline
    self.video_player.set_state(gst.STATE_PLAYING)       # start video stream

    bus = self.video_player.get_bus()
    bus.add_signal_watch()
    bus.connect("message", self.on_message)
    bus.enable_sync_message_emission()
    bus.connect("sync-message::element", self.on_sync_message)

    def on_message(self, bus, message):
        """ Gst message bus. Closes the pipeline in case of error or EOS (end of stream) message """
        t = message.type
        if t == gst.MESSAGE_EOS:
            print "MESSAGE EOS"
            self.video_player.set_state(gst.STATE_NULL)
        elif t == gst.MESSAGE_ERROR:
            print "MESSAGE ERROR"
            err, debug = message.parse_error()
            print "Error: %s" % err, debug
            self.video_player.set_state(gst.STATE_NULL)
                    
    def on_sync_message(self, bus, message):
        """ Set up the Webcam <--> GUI messages bus """
        if message.structure is None:
            return
        message_name = message.structure.get_name()
        if message_name == "prepare-xwindow-id":
            # Assign the viewport
            imagesink = message.src
            imagesink.set_property("force-aspect-ratio", True)
            imagesink.set_xwindow_id(self.movie_window.window.xid) # Sending video stream to gtk DrawingArea

def take_snapshot(self):
    """ Capture a snapshot from DrawingArea and save it into a image file """
    drawable = self.movie_window.window
    # self.movie_window is of type gtk.DrawingArea()
    colormap = drawable.get_colormap()
    pixbuf = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, 0, 8, *drawable.get_size())
    pixbuf = pixbuf.get_from_drawable(drawable, colormap, 0,0,0,0, *drawable.get_size()) 
    pixbuf = pixbuf.scale_simple(self.W, self.H, gtk.gdk.INTERP_HYPER) # resize
    # We resize from actual window size to wanted resolution
    #  gtk.gdk.INTER_HYPER is the slowest and highest quality reconstruction function
    # More info here : http://developer.gnome.org/pygtk/stable/class-gdkpixbuf.html#method-gdkpixbuf--scale-simple
    filename = 'snap.jpg'
    filepath = relpath(filename)
    pixbuf.save(filename, self.snap_format)
