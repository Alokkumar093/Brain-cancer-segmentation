# brain-tumor-segmentation

## Project Overview

This project utilizes a pre-trained U-Net model to segment brain tumors from MRI scans (FLAIR and T1CE). A FastAPI backend handles API calls, while a frontend interface provides easy upload and visualization.

## Features

-   Brain tumor segmentation using MRI scans (FLAIR and T1CE)
-   FastAPI backend for efficient API calls
-   Frontend interface for easy upload and visualization
-   GIF output displaying segmented tumor


## Requirements

*   Python 3.10
*   FastAPI
*   Uvicorn
*   NumPy
*   NiBabel
*   ImageIO
*   Matplotlib
*   Scikit-image
*   TensorFlow


## Installation

1.  Clone the repository: `git clone git@github.com:S4M0707/brain-tumor-segmentation.git`
2.  Install dependencies: `pip install -r requirements.txt`
3.  Run the FastAPI server: `uvicorn main:app`


## Usage

2.  Upload FLAIR and T1CE MRI scans.
3.  Click "Submit" to segment the tumor.
4.  Visualize the segmented tumor in the GIF output.


## Contributing

Contributions are welcome! Please fork the repository, make changes, and submit a pull request.


## License

This project is licensed under the MIT License.

## Future Work

*   Improve model accuracy
*   Add support for other MRI modalities
*   Integrate with clinical systems


## Project Structure

*   `main.py`: FastAPI application
*   `utils/brain_tumor.py`: Brain tumor segmentation model
*   `index.html`: Frontend interface
*   `requirements.txt`: Dependencies