### How to organise your workspace for the course: general tips

Here we list several tips for better organisation of your workspace and coding infrastructure for the course. We encourage you to follow these tips when working with course materials and/or submitting your final project. Also, we find these tips generally useful for any programming work conducted in the MLT program.

#### Coding on mlt-gpu from your local machine
You can run your scripts on mlt-gpu by running everything on your local computer. First, enter the following command in the terminal on your local machine: ``ssh -p 62266 xilini@mltgpu.flov.gu.se -L8000:localhost:8000``. This should connect you to mlt-gpu and book a port (8000) on the server for you. The port can have any number, which is not already used by someone. Then, on your local machine, open ``127.0.0.1:8000`` in your browser. You should see the starting window for Jupyter Lab (given that you installed it on mlt-gpu). If required, enter the login token, you can access it in the terminal where you ran your ssh command. That's it! Now you can navigate mlt-gpu from your local machine and run any code remotely.

#### Use virtual environments
Virtual environments allow you to avoid any conflicts between different versions of your project and would really help your TAs to navigate your code and test it. For more, you can read [this thread](https://stackoverflow.com/questions/41972261/what-is-a-virtualenv-and-why-should-i-use-one) (installing environments with pip) or [this one](https://towardsdatascience.com/introduction-to-conda-virtual-environments-eaea4ac84e28) (installing environments with conda).

#### Use Jupyter Lab for interactive work
Use Jupyter Lab when you want to inspect individual parts of the code and/or want to 'interact' more with your code. Python scripts (.py) should be used when you submit your project, but script testing, discussions with TAs regarding the code should be accompanied by an interactive notebook. It is generally more convenient to work this way since your TA (and you as well) could run individual parts of the code in isolation when required.

#### Use screen/tmux
You often want to train your model overnight without breaking/stopping the training scripts. To achieve this, use [screen](https://linuxize.com/post/how-to-use-linux-screen/) or [tmux](https://linuxize.com/post/getting-started-with-tmux/). They are already installed on mlt-gpu and accessible to everyone. Start your scripts in screen/tmux session (you have to create one first), and your code will run safely in the background even when you log out. Beforehand, we suggest you to test your scripts on a small subset of your data to make sure that training, validation, evaluation and testing loops are working well. Otherwise, when you check your screen/tmux session the next day, you can see an unexpected error coming from your code, which means that you will have to re-run your scripts.
