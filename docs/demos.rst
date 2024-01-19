Demos
================

Multi-object using ROS and DOPE with RGB and Depth
----------

The demo uses RGB and Depth from a RealSense camera,
and poses published by DOPE. You can use your own configuration, but you will
need to adapt the ``configs/multiobject_with_dope.yaml`` file.
You can also download a ROS bag from `here <https://leeds365-my.sharepoint.com/personal/scsrp_leeds_ac_uk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fscsrp%5Fleeds%5Fac%5Fuk%2FDocuments%2FResearch%2Fsimple%5Fhope%5Fscene%5Fwith%5Fdope%5Fdetections%2Ebag&parent=%2Fpersonal%2Fscsrp%5Fleeds%5Fac%5Fuk%2FDocuments%2FResearch&ga=1>`_.

If you wish to use the ROS bag, you can play it back in a loop like so:

.. code::

    rosbag play -l ~/path/to/simple_hope_scene.bag

From there you can run the demo like so:

.. code::

    cd ~/diff-dope
    python3 examples/ros_with_dope.py
