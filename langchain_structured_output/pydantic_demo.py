from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):

    name: str = 'Rabiul Islam' # This name work as a default value.
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default = 5, description='A decimal value representing the cgpa of the student')

new_student = {'name': 'Pappu Akondo', 'age': '32', 'email': 'abc@gmail.com', 'cgpa': 9}

student = Student(**new_student)

student_dict = dict(student)

print(student_dict['age'])

student_json = student.model_dump_json()