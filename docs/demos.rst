Demos
================

Multi-object using ROS and DOPE with RGB and Depth
----------

The demo uses RGB and Depth from a RealSense camera,
and poses published by DOPE. You can use your own configuration, but you will
need to adapt the ``configs/multiobject_with_dope.yaml`` file.
You can also download a ROS bag from `here <https://www.TODO.com>`_.

If you wish to use the ROS bag, you can play it back in a loop like so:

.. code::

    rosbag play -l ~/path/to/simple_hope_scene.bag

From there you can run the demo like so:

.. code::

    cd ~/diff-dope
    python3 examples/ros_with_dope.py
