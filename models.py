from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class User(db.Model):
    """User model with role-based access"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default='user')  # user, admin, super_admin
    department = db.Column(db.String(100), default='')
    phone = db.Column(db.String(20), default='')
    profile_image = db.Column(db.Text, default='')
    status = db.Column(db.String(20), default='active')  # active, inactive, suspended
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self, include_sensitive=False):
        data = {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'role': self.role,
            'department': self.department,
            'phone': self.phone,
            'profile_image': self.profile_image,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }
        if include_sensitive:
            data['password_hash'] = self.password_hash
        return data
    
    def is_super_admin(self):
        return self.role == 'super_admin'
    
    def is_admin(self):
        return self.role in ['admin', 'super_admin']


class Person(db.Model):
    """Person with face embeddings for recognition"""
    __tablename__ = 'persons'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    name = db.Column(db.String(100), nullable=False, index=True)
    embedding = db.Column(db.LargeBinary, nullable=False)
    embedding_dim = db.Column(db.Integer, nullable=False)
    photos_count = db.Column(db.Integer, default=0)
    status = db.Column(db.String(20), default='active')
    enrollment_date = db.Column(db.DateTime, default=datetime.utcnow)
    enrolled_by = db.Column(db.String(120), nullable=True)
    
    user = db.relationship('User', backref='person_record', foreign_keys=[user_id])
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'name': self.name,
            'photos_count': self.photos_count,
            'status': self.status,
            'enrollment_date': self.enrollment_date.isoformat() if self.enrollment_date else None,
            'enrolled_by': self.enrolled_by
        }


class EnrollmentRequest(db.Model):
    """Enrollment requests from users"""
    __tablename__ = 'enrollment_requests'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False, index=True)
    phone = db.Column(db.String(20), default='')
    department = db.Column(db.String(100), default='')
    password_hash = db.Column(db.String(255), nullable=False)
    images = db.Column(db.JSON, nullable=False)
    status = db.Column(db.String(20), default='pending')  # pending, approved, rejected
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)
    processed_at = db.Column(db.DateTime, nullable=True)
    processed_by = db.Column(db.String(120), nullable=True)
    rejection_reason = db.Column(db.Text, nullable=True)
    
    def to_dict(self, include_images=False):
        data = {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'phone': self.phone,
            'department': self.department,
            'status': self.status,
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
            'processed_by': self.processed_by,
            'rejection_reason': self.rejection_reason
        }
        if include_images:
            data['images'] = self.images
        return data


class Attendance(db.Model):
    """Attendance records"""
    __tablename__ = 'attendance'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    person_id = db.Column(db.Integer, db.ForeignKey('persons.id'), nullable=True)
    name = db.Column(db.String(100), nullable=False, index=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    date = db.Column(db.Date, default=datetime.utcnow().date, index=True)
    confidence = db.Column(db.Float, default=0.0)
    image = db.Column(db.Text, nullable=True)
    location = db.Column(db.String(255), default='')
    status = db.Column(db.String(20), default='present')  # present, late, absent
    
    user = db.relationship('User', backref='attendances', foreign_keys=[user_id])
    person = db.relationship('Person', backref='attendances', foreign_keys=[person_id])
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'person_id': self.person_id,
            'name': self.name,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'date': self.date.isoformat() if self.date else None,
            'confidence': round(self.confidence, 3),
            'location': self.location,
            'status': self.status
        }


class SystemLog(db.Model):
    """System activity logs for super-admin monitoring"""
    __tablename__ = 'system_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    action = db.Column(db.String(100), nullable=False, index=True)
    entity_type = db.Column(db.String(50), nullable=True)
    entity_id = db.Column(db.Integer, nullable=True)
    details = db.Column(db.JSON, nullable=True)
    ip_address = db.Column(db.String(50), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    user = db.relationship('User', backref='logs', foreign_keys=[user_id])
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'user_name': self.user.name if self.user else 'System',
            'action': self.action,
            'entity_type': self.entity_type,
            'entity_id': self.entity_id,
            'details': self.details,
            'ip_address': self.ip_address,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


class SystemSettings(db.Model):
    """System-wide settings managed by super-admin"""
    __tablename__ = 'system_settings'
    
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(100), unique=True, nullable=False, index=True)
    value = db.Column(db.Text, nullable=True)
    value_type = db.Column(db.String(20), default='string')  # string, int, bool, json
    description = db.Column(db.Text, nullable=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = db.Column(db.String(120), nullable=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'key': self.key,
            'value': self.value,
            'value_type': self.value_type,
            'description': self.description,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'updated_by': self.updated_by
        }


# Create indexes for better performance
db.Index('idx_attendance_date', Attendance.date.desc())
db.Index('idx_attendance_user', Attendance.user_id, Attendance.date.desc())
db.Index('idx_persons_status', Person.status)
db.Index('idx_users_role', User.role)
db.Index('idx_enrollment_status', EnrollmentRequest.status)
db.Index('idx_logs_timestamp', SystemLog.timestamp.desc())
