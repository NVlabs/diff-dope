Demos
================

Multi-object using ROS and DOPE with RGB and Depth
--------------------------------------------------

The demo uses RGB and Depth from a RealSense camera, poses published by DOPE,
and segments the image using
`segment-anything <https://github.com/facebookresearch/segment-anything>`_.
The demo comes as a catkin package under ``diffdope_ros/``.

Say you cloned the diff-dope repo under your home directory (i.e., ``~/diff-dope``),
and you have a catkin workspace under your home directory too (i.e., ``~/catkin_ws``),
you can create a symlink of the package there:

.. code::

    cd ~/catkin_ws/src
    ln -s ~/diff-dope/diffdope_ros .

You can ``catkin_make`` under ``~/catkin_ws`` now to build the package.
You can, of course, move the package there instead of creating a symlink.

The demo uses a configuration under
``diffdope_ros/config/multiobject_with_dope.yaml`` which uses DOPE for initial
pose estimation, and dictates topics for RGB and Depth (assuming a RealSense
sensor). It also uses a camera info topic to retrieve camera intrinsic
parameters as well as the image dimensions.

You can also download a ROS bag from `here <https://leeds365-my.sharepoint.com/:u:/g/personal/scsrp_leeds_ac_uk/Ec-TbyOr1QVIt6NQQP7E4pABkEUmaEGByVjLHugY7Als_A?e=JES96n>`_
to play with this demo, without the need of a real sensor.

If you wish to use the ROS bag, you can play it back in a loop like so:

.. code::

    rosbag play -l ~/path/to/simple.bag


Install diffdope, under the root diffdope directory:

.. code::

    pip install -r requirements.txt
    pip install -e .


You need to install
`segment-anything <https://github.com/facebookresearch/segment-anything>`_
and download the checkpoints.

.. code::

    pip install git+https://github.com/facebookresearch/segment-anything.git

Then head to the
`model checkpoints section <https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints>`_,
and download any of the model checkpoints. Place it somewhere on your
computer and update the ``segment_anything_checkpoint_path`` of the config file
to let it know where to find it. If you want to test the segmentation
functionality independently, you can run the following Python script:
``diffdope_ros/scripts/segmentator.py``.

Now run DOPE using:

.. code::

   roslaunch dope dope.launch


Make sure DOPE detects the objects and that the topics you expect are being
published. From there you can run the ``refine.launch`` demo like so:

.. code::

    roslaunch diffdope_ros server.py # To start the actionlib server

    # Refine pose for individual object
    roslaunch diffdope_ros refine.launch object_name:=bbq_sauce
    roslaunch diffdope_ros refine.launch object_name:=alphabet_soup
    roslaunch diffdope_ros refine.launch object_name:=mustard

    # or don't pass object_name to refine the pose of all the objects in the config
    roslaunch diffdope_ros refine.launch

The above names are derived from the config file, ``config/multiobject_with_dope.yaml``.
The launch files pass this file to the server and refine scripts.

The ``refine.launch`` is just doing a single refinement and exit. There is
also the ``refine_continuous.launch`` that will do continuous refinement
and publish the refined pose to a topic ``/diffdope_object_name``.

.. code::

    roslaunch diffdope_ros server.py # To start the actionlib server

    # Refine pose for individual object
    roslaunch diffdope_ros refine_continuous.launch object_name:=bbq_sauce
    roslaunch diffdope_ros refine_continuous.launch object_name:=alphabet_soup
    roslaunch diffdope_ros refine_continuous.launch object_name:=mustard

    # or don't pass object_name to refine the pose of all the objects in the config
    roslaunch diffdope_ros refine_continuous.launch


Parameters
--------------------------------------------------

Please inspect the ``config/multiobject_with_dope.yaml`` file to see certain
parameters. For example, by default this demo will produce videos of the
optimisation, however you can turn this off through the config file
to speed things up. You can also adjust certain optimisation parameters from
the config file. Some parameters related to the ROS demo below:

+-----------------+-----------------+
| Parameter | Explanation        |
+=================+=================+
| segment_anything_checkpoint_path | path to the checkpoint file for segment anything. |
+-----------------+-----------------+
| save_video | whether to save a video visualizing the diffdope optimization. |
+-----------------+-----------------+
| max_saved_videos_per_run | This will dictate how many videos will be generated per run, to avoid generating too many when running the continuous version. |
+-----------------+-----------------+

Dealing with DOPE and model coordinate frames
--------------------------------------------------

Please note the following important details when you try to use a new object
and pose from DOPE:

* DOPE pose output may not match the coordinate frame used in the 3D model of
  the object you wish to use. In this case, you need to apply a static
  transformation to bring the DOPE pose output to match the one used in your 3D
  model. DOPE provides a way in the config file (``model_transforms``) to define such transformation
  per object. For more details on this subject, please read `this <https://github.com/NVlabs/Deep_Object_Pose/issues/346>`_.
* The scaling of the object is important. We suggest that you scale your 3D object
  in Blender to bring it closer to the scale of the examples. For example,
  the HOPE objects as downloaded from the official repository, we had to scale them
  by a factor of 10. Although a parameter to scale the 3D object in the config
  is available, we had difficulties to get it to work properly and found better
  luck by manually scaling the 3D object in Blender. You can import a reference
  object (like the BBQ Sauce model we provide) in Blender to see the scale.
