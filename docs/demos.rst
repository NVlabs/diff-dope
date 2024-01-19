Demos
================

Multi-object using ROS and DOPE with RGB and Depth
--------------------------------------------------

The demo uses RGB and Depth from a RealSense camera,
and poses published by DOPE. The demo comes as a catkin package
under `examples/`.

Say you cloned the diff-dope repo under your home directory (i.e., `~/diff-dope`),
and you have a catkin workspace under your home directory too (i.e., `~/catkin_ws`),
you can create a symlink of the package there:

.. code::

    cd ~/catkin_ws/src
    ln -s ~/diff-dope/examples/diffdope_ros .

You can `catkin_make` under `~/catkin_ws` now to build the package.
You can, of course, move the package there instead of creating a symlink.

The demo uses a configuration under
`diffdope_ros/config/multiobject_with_dope.yaml` which uses DOPE for initial
pose estimation, and dictates topics for RGB and Depth (assuming a RealSense
sensor).

You can also download a ROS bag from `here
<https://leeds365-my.sharepoint.com/personal/scsrp_leeds_ac_uk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fscsrp%5Fleeds%5Fac%5Fuk%2FDocuments%2FResearch%2Fsimple%5Fhope%5Fscene%5Fwith%5Fdope%5Fdetections%2Ebag&parent=%2Fpersonal%2Fscsrp%5Fleeds%5Fac%5Fuk%2FDocuments%2FResearch&ga=1>`_
to play with this demo, without the need of a real sensor.

If you wish to use the ROS bag, you can play it back in a loop like so:

.. code::

    rosbag play -l ~/path/to/simple_hope_scene_with_dope_detections.bag

From there you can run the demo like so:

.. code::

    rosrun diffdope_ros server.py # To start the actionlib server

    # Refine pose for individual object
    rosrun diffdope_ros refine.py object_name:=some_object_name1
    rosrun diffdope_ros refine.py object_name:=some_object_name2

    # Refine pose for all objects
    rosrun diffdope_ros refine.py

The possible object names for this demo are: ``bbq_sauce, orange_juice, mustard``
(as defined in the config file, ``diffdope_ros/config/multiobject_with_dope.yaml``).
