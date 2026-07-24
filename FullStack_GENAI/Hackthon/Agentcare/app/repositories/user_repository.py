from sqlalchemy.orm import Session

from app.models.user import User


class UserRepository:

    def create_user(
        self,
        db: Session,
        user: User
    ) -> User:
        """
        Save a new user to the database.
        """
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    def get_user_by_email(
        self,
        db: Session,
        email: str
    ) -> User | None:
        """
        Fetch a user by email.
        """
        return (
            db.query(User)
            .filter(User.email == email)
            .first()
        )

    def get_user_by_id(
        self,
        db: Session,
        user_id: int
    ) -> User | None:
        """
        Fetch a user by ID.
        """
        return (
            db.query(User)
            .filter(User.id == user_id)
            .first()
        )


user_repository = UserRepository()