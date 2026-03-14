import sys
# sys library is used to get the current exception information

# this file contains the custom exception class which will be used to handle exceptions in the project, it will provide detailed error messages with the file name and line number where the exception occurred, this will help in debugging the code and finding the root cause of the exception

def error_message_detail(error,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info() # exc_info() returns a tuple of three values: (type, value, traceback) , type and value are not used in this function, so we can ignore them by using _ , and exc_tb is the traceback object which contains information about the exception
    file_name = exc_tb.tb_frame.f_code.co_filename # tb_frame is the frame object, f_code is the code object, co_filename is the filename
    line_number = exc_tb.tb_lineno # tb_lineno is the line number where the exception occurred
    error_message = f"Error occurred in script: [{file_name}] at line number: [{line_number}] with error message: [{error}]"
    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message) # call the constructor of the parent class (Exception) to initialize the error message
        self.error_message = error_message_detail(error_message,error_detail) # call the error_message_detail function to get the detailed error message

    def __str__(self):
        return self.error_message # return the detailed error message when the exception is printed