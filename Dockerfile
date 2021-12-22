FROM python:3.9

# Add McBackend
COPY . /mcbackend
RUN ls -a /mcbackend && pip install -e /mcbackend

# Add Streamlit app
COPY arviz-server/requirements.txt /arviz-server/requirements.txt
COPY arviz-server/app.py /arviz-server/app.py
RUN pip install -r /arviz-server/requirements.txt

# Startup configuration
EXPOSE 8501
ENTRYPOINT ["streamlit", "run"]
CMD ["arviz-server/app.py"]
