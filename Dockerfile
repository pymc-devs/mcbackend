FROM python:3.9

# Add McBackend
COPY . /mcbackend
RUN ls -a /mcbackend && pip install -e /mcbackend

# Add Streamlit app
COPY mcbackend-server/requirements.txt /mcbackend-server/requirements.txt
COPY mcbackend-server/app.py /mcbackend-server/app.py
RUN pip install -r /mcbackend-server/requirements.txt

# Startup configuration
EXPOSE 8501
ENTRYPOINT ["streamlit", "run"]
CMD ["mcbackend-server/app.py"]
