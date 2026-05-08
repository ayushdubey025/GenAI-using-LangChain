from typing import TypedDict

class Student(TypedDict):
    name: str
    marks: int

new_student: Student= {"name": "Ranvir", "marks": 84}

print(new_student)