from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.auth.password import hash_password
from app.models.user import User
from app.repositories.user_repository import user_repository
from app.schemas.user import UserCreate


class UserService:

    def register_user(
        self,
        db: Session,
        user_data: UserCreate
    ) -> User:

        # Check if email already exists
        existing_user = user_repository.get_user_by_email(
            db,
            user_data.email
        )

        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered."
            )

        # Create User model
        new_user = User(
            name=user_data.name,
            email=user_data.email,
            password_hash=hash_password(user_data.password),
            role="PATIENT"
        )

        # Save user
        return user_repository.create_user(
            db,
            new_user
        )


user_service = UserService()