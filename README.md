# Attendance-System-Using-Face-Recognition-02
#Create virtual environment
#install requirements.txt
#Install CMake manually
#inatall dlib manually
https://pysource.com/2019/03/20/how-to-install-dlib-for-python-3-on-windows/
https://pypi.org/simple/dlib/
#python -m pip install https://files.pythonhosted.org/packages/0e/ce/f8a3cff33ac03a8219768f0694c5d703c8e037e6aba2e865f9bae22ed63c/dlib-19.8.1-cp36-cp36m-win_amd64.whl#sha256=794994fa2c54e7776659fddb148363a5556468a6d5d46be8dad311722d54fcf

#  
#Git clone Developer branch steps

1#Generate SSH key
ssh-keygen -t rsa -b 4096 -C "bhavindjaviya@gmail.com"
2# show SSH key 
cat ~/.ssh/id_rsa.pub
3#copy key and paste to "Setings>SSH and GPG key>New SSH key >Add SSH Key"
4# check status of authentication
ssh -T git@github.com
5#Message:Hi your-username! You've successfully authenticated, but GitHub does not provide shell access.
6#copy SSH repository link:
7#clone
git clone -b Developer git@github.com:bhavin108/Attendance-System-Using-Face-Recognition-02.git


mkvirtualenv myenv --python=python3.6

workon myenv
pip install django

#update settings.py:
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'static')

#collect static files
python manage.py collectstatic

Run the database migrations
python manage.py migrate

#set all path

and wsgi file copy.
