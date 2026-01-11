import sys  # Import the sys module, which provides access to system-specific parameters and functions, like error details.
from src.logger import logging  # Import the logging object from the logger module located in the src package.
def error_message_detail(error, error_detail: sys):  # Define a function named error_message_detail that takes an error message and a sys object for error details.
    _, _, exc_tb = error_detail.exc_info()  # Get the exception traceback info from sys.exc_info(), which returns a tuple; we ignore the first two items and take the traceback object (exc_tb).
    file_name = exc_tb.tb_frame.f_code.co_filename  # Extract the filename where the error occurred from the traceback's frame.
    line_number = exc_tb.tb_lineno  # Extract the line number where the error occurred from the traceback.
    error_message = f"Error occurred in script: {file_name} at line number: {line_number} with error message: {str(error)}"  # Create a formatted string (f-string) that combines the file name, line number, and error message.
    return error_message  # Return the formatted error message string to the caller.

class CustomException(Exception):  # Define a custom exception class that inherits from Python's built-in Exception class, allowing us to create our own error types.
    def __init__(self, error_message, error_detail: sys):  # Define the initializer method (__init__) for the class, which runs when creating a new instance. It takes an error message and sys object.
        super().__init__(error_message)  # Call the parent Exception class's __init__ method with the error message to set up the base exception.
        self.error_message = error_message_detail(error_message, error_detail=error_detail)  # Call the error_message_detail function and store its result in the object's error_message attribute.

    def __str__(self):  # Define the __str__ method, which returns a string representation of the exception (used when printing or converting to string).
        return self.error_message  # Return the stored error message string.
    