app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello World"} 

@app.post("/files/")
async def create_file(file: bytes = File()):
    print(file)
    return {"file_size": len(file)}