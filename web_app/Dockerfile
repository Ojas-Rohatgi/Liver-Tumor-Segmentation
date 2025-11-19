# Start from a slim Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first and install dependencies
# This caches the installed packages, so it doesn't re-install every time you change code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your application code into the container
COPY . .

# Expose the port Gunicorn will run on
EXPOSE 5000

# This is the default command.
# The docker-compose file will override this for the 'worker' service.
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]