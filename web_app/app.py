# app.py
from flask import Flask, request, render_template, jsonify, send_from_directory
from pathlib import Path
import uuid
import sys
from celery.result import AsyncResult

# --- Import our new task ---
# This imports the 'celery' object and the 'run_inference_task'
try:
    from tasks import celery, run_inference_task
except ImportError:
    print("Error: Could not import tasks.py. Make sure it is in the same directory.")
    sys.exit(1)

# --- Configuration ---
app = Flask(__name__, static_folder='.', template_folder='.')


# No longer need model paths or process tracking here!


@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start the inference task."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Create a unique folder for this request
        request_id = str(uuid.uuid4())
        upload_dir = Path.cwd() / "uploads" / request_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        input_path = upload_dir / "volume.nii"
        file.save(str(input_path))

        # Define output paths
        output_dir = Path.cwd() / "static" / request_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_2d_path = output_dir / "prediction_2d.png"
        output_3d_path = output_dir / "prediction_3d.html"

        # --- Launch inference as a Celery task ---
        try:
            # We pass paths as strings, as they are easier to serialize for Celery
            task = run_inference_task.delay(
                str(input_path),
                str(output_2d_path),
                str(output_3d_path)
            )

            print(f"Started inference task for request {request_id} with Task ID {task.id}")

            # Return a response immediately with the task_id
            return jsonify({
                'message': 'Inference started.',
                'request_id': request_id,
                'task_id': task.id  # <-- NEW: Send task_id, not pid
            }), 202  # 202 Accepted

        except Exception as e:
            print(f"Error starting celery task: {e}")
            return jsonify({'error': f'Error starting inference: {e}'}), 500


@app.route('/status/<task_id>', methods=['GET'])
def get_status(task_id):
    """Check the status of a running Celery task."""

    # Get the task result from the Redis backend
    task_result = AsyncResult(task_id, app=celery)

    if task_result.state == 'PENDING':
        # Task is waiting for a worker
        return jsonify({'status': 'PENDING', 'message': 'Task is queued...'}), 200

    elif task_result.state == 'PROGRESS':
        # Our task is running and providing progress
        return jsonify({'status': 'RUNNING',
                        'message': f"Running {task_result.info.get('stage')}... ({task_result.info.get('current')}/{task_result.info.get('total')})"}), 200

    elif task_result.state == 'SUCCESS':
        # Task finished successfully
        return jsonify({'status': 'COMPLETE', 'message': 'Inference complete.'}), 200

    elif task_result.state == 'FAILURE':
        # Task failed
        # task_result.info will contain the exception
        error_message = str(task_result.info)
        print(f"Task {task_id} failed with error: {error_message}")
        return jsonify({'status': 'ERROR', 'message': error_message}), 500

    else:
        # Other states
        return jsonify({'status': task_result.state, 'message': 'Task in unknown state.'}), 200


@app.route('/results/<path:request_id>/<path:filename>')
def get_result_file(request_id, filename):
    """Serve the generated plot files."""
    file_dir = Path.cwd() / "static" / str(request_id)
    return send_from_directory(file_dir, filename)


if __name__ == '__main__':
    # Ensure static and uploads dirs exist
    (Path.cwd() / "static").mkdir(exist_ok=True)
    (Path.cwd() / "uploads").mkdir(exist_ok=True)

    # This is still fine for local testing,
    # but for production, you'll use Gunicorn
    app.run(debug=True, host='0.0.0.0', port=5000)