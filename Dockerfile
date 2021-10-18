FROM python:3.9

# Create user
RUN useradd -ms /bin/bash newuser
USER newuser
ENV PATH=$PATH:"/home/newuser/.local/bin"
RUN /usr/local/bin/python -m pip install --upgrade pip

RUN pip install numpy==1.20.2 flake8 pytest==6.2.3