# Use an official CentOS image as a parent image
FROM centos:7

# Set the working directory to /app
WORKDIR /app

# Copy your Spark application files to the container
COPY . /app

# Install dependencies
# RUN yum install epel-release -y
# RUN yum update -y
# Python 
RUN yum install python3 -y
RUN yum install python3-pip  -y
# # Java 
# RUN yum install -y java-1.8.0-openjdk
# # Set environment variables for Java
# ENV JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk
# ENV PATH=$PATH:$JAVA_HOME/bin
RUN yum install java -y

RUN pip3 install -r requirements.txt



# # Download and extract Apache Spark
# ADD http://apache.mirrors.pair.com/spark/spark-3.1.2/spark-3.1.2-bin-hadoop2.7.tgz /app
# RUN tar -xvzf spark-3.1.2-bin-hadoop2.7.tgz && rm spark-3.1.2-bin-hadoop2.7.tgz && mv spark-3.1.2-bin-hadoop2.7 spark

# # Set Spark home and add Spark binaries to PATH
# ENV SPARK_HOME /app/spark
# ENV PATH $PATH:$SPARK_HOME/bin



# Run your Spark application
CMD ["spark-submit", "training.py", "&&", "spark-submit", "testing.py"]
# ENTRYPOINT ["spark-submit", "testing.py"]

