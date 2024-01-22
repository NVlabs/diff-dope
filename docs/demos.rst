Demos
================

Multi-object using ROS and DOPE with RGB and Depth
--------------------------------------------------

The demo uses RGB and Depth from a RealSense camera, poses published by DOPE,
and segments the image using
`segment-anything <https://github.com/facebookresearch/segment-anything>`_.
The demo comes as a catkin package under ``examples/``.

Say you cloned the diff-dope repo under your home directory (i.e., ``~/diff-dope``),
and you have a catkin workspace under your home directory too (i.e., ``~/catkin_ws``),
you can create a symlink of the package there:

.. code::

    cd ~/catkin_ws/src
    ln -s ~/diff-dope/examples/diffdope_ros .

You can ``catkin_make`` under ``~/catkin_ws`` now to build the package.
You can, of course, move the package there instead of creating a symlink.

The demo uses a configuration under
``diffdope_ros/config/multiobject_with_dope.yaml`` which uses DOPE for initial
pose estimation, and dictates topics for RGB and Depth (assuming a RealSense
sensor).

You can also download a ROS bag from `here (simple scene)
<https://leeds365-my.sharepoint.com/personal/scsrp_leeds_ac_uk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fscsrp%5Fleeds%5Fac%5Fuk%2FDocuments%2FResearch%2Fsimple%5Fhope%5Fscene%5Fwith%5Fdope%5Fdetections%2Ebag&parent=%2Fpersonal%2Fscsrp%5Fleeds%5Fac%5Fuk%2FDocuments%2FResearch&ga=1>`_
or `here (cluttered scene)
<https://leeds365-my.sharepoint.com/:u:/g/personal/scsrp_leeds_ac_uk/EUO5a2GfZRFOrueUzRbkLSwBZD3WoTsm5MP8hXeF0AYAEw?e=3YOqr6>`_
to play with this demo, without the need of a real sensor.

If you wish to use the ROS bag, you can play it back in a loop like so:

.. code::

    rosbag play -l ~/path/to/simple_hope_scene_with_dope_detections.bag
    # or
    rosbag play -l ~/path/to/cluttered_hope_scene_with_dope_detections.bag


You need to install
`segment-anything <https://github.com/facebookresearch/segment-anything>`_
and download the weights of the model.

.. code::

    pip install git+https://github.com/facebookresearch/segment-anything.git


Then head to the
`model checkpoints section <https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints>`_,
and download any of the model checkpoints. Place it somewhere on your
computer and update the ``segment_anything_checkpoint_path`` of the config file
to let it know where to find it. If you want to test the segmentation
functionality independently, you can run the following Python script:
``diffdope_ros/scripts/segmentator.py``.

From there you can run the demo like so:

.. code::

    rosrun diffdope_ros server.py # To start the actionlib server

    # Refine pose for individual object
    rosrun diffdope_ros refine.py object_name:=bbq_sauce
    rosrun diffdope_ros refine.py object_name:=orange_juice
    rosrun diffdope_ros refine.py object_name:=mustard

    # or don't pass object_name to refine the pose of all the objects in the config
    rosrun diffdope_ros refine.py

The possible object names for this demo are: ``bbq_sauce, orange_juice, mustard``
(as defined in the config file, ``diffdope_ros/config/multiobject_with_dope.yaml``).
