import os
import logging
from datetime import datetime, timedelta
from functools import wraps
import base64
import numpy as np

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_from_directory
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash
from dotenv import load_dotenv

from models import db, User, Person, EnrollmentRequest, Attendance, SystemLog, SystemSettings
from services.face_service import face_service
from services.faiss_service import faiss_service

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'jwt-secret-key-change-in-production')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///attendance.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize extensions
db.init_app(app)
jwt = JWTManager(app)


#  HELPER FUNCTIONS 

def log_activity(action, user_id=None, entity_type=None, entity_id=None, details=None):
    """Log system activity"""
    try:
        log = SystemLog(
            user_id=user_id,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            details=details,
            ip_address=request.remote_addr
        )
        db.session.add(log)
        db.session.commit()
    except Exception as e:
        logger.error(f"Failed to log activity: {e}")


def role_required(*roles):
    """Decorator to check user role"""
    def decorator(f):
        @wraps(f)
        @jwt_required()
        def decorated_function(*args, **kwargs):
            current_user_email = get_jwt_identity()
            user = User.query.filter_by(email=current_user_email).first()
            
            if not user or user.role not in roles:
                return jsonify({'error': 'Insufficient permissions'}), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


# ===================== AUTHENTICATION ROUTES =====================

@app.route('/')
def index():
    """Landing page"""
    return render_template('index.html')


@app.route('/login')
def login_page():
    """Login page"""
    return render_template('login.html')


@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login endpoint"""
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
        user = User.query.filter_by(email=email).first()
        
        if not user or not user.check_password(password):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        if user.status != 'active':
            return jsonify({'error': 'Account is not active'}), 403
        
        # Update last login
        user.last_login = datetime.utcnow()
        db.session.commit()
        
        # Create access token
        access_token = create_access_token(identity=email)
        
        log_activity('login', user.id)
        
        return jsonify({
            'access_token': access_token,
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500


@app.route('/api/auth/register', methods=['POST'])
def register():
    """User registration with enrollment request"""
    try:
        data = request.get_json()
        
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')
        phone = data.get('phone', '')
        department = data.get('department', '')
        images = data.get('images', [])
        
        if not all([name, email, password]):
            return jsonify({'error': 'Name, email, and password required'}), 400
        
        if len(images) < 3:
            return jsonify({'error': 'At least 3 images required'}), 400
        
        # Check if user already exists
        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already registered'}), 409
        
        # Check if enrollment request exists
        if EnrollmentRequest.query.filter_by(email=email, status='pending').first():
            return jsonify({'error': 'Enrollment request already pending'}), 409
        
        # Create enrollment request
        enrollment_req = EnrollmentRequest(
            name=name,
            email=email,
            phone=phone,
            department=department,
            password_hash=generate_password_hash(password),
            images=images,
            status='pending'
        )
        
        db.session.add(enrollment_req)
        db.session.commit()
        
        log_activity('enrollment_request_submitted', None, 'enrollment_request', enrollment_req.id)
        
        return jsonify({
            'message': 'Enrollment request submitted successfully',
            'request_id': enrollment_req.id
        }), 201
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Registration error: {e}")
        return jsonify({'error': 'Registration failed'}), 500


# ===================== USER DASHBOARD =====================

@app.route('/dashboard')
@jwt_required(optional=True)
def dashboard():
    """User dashboard page"""
    return render_template('user_dashboard.html')


@app.route('/api/user/stats', methods=['GET'])
@jwt_required()
def user_stats():
    """Get user statistics"""
    try:
        current_user_email = get_jwt_identity()
        user = User.query.filter_by(email=current_user_email).first()
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get attendance stats
        today = datetime.utcnow().date()
        this_month_start = datetime(today.year, today.month, 1)
        
        total_days = Attendance.query.filter(
            Attendance.user_id == user.id,
            Attendance.date >= this_month_start
        ).distinct(Attendance.date).count()
        
        present_days = Attendance.query.filter(
            Attendance.user_id == user.id,
            Attendance.date >= this_month_start,
            Attendance.status == 'present'
        ).distinct(Attendance.date).count()
        
        late_days = Attendance.query.filter(
            Attendance.user_id == user.id,
            Attendance.date >= this_month_start,
            Attendance.status == 'late'
        ).distinct(Attendance.date).count()
        
        # Calculate attendance percentage
        working_days = (today - this_month_start.date()).days + 1
        attendance_percentage = (present_days / working_days * 100) if working_days > 0 else 0
        
        return jsonify({
            'total_days': total_days,
            'present_days': present_days,
            'late_days': late_days,
            'attendance_percentage': round(attendance_percentage, 1),
            'working_days': working_days
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching user stats: {e}")
        return jsonify({'error': 'Failed to fetch statistics'}), 500


@app.route('/api/user/attendance/history', methods=['GET'])
@jwt_required()
def user_attendance_history():
    """Get user attendance history"""
    try:
        current_user_email = get_jwt_identity()
        user = User.query.filter_by(email=current_user_email).first()
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        
        attendance_records = Attendance.query.filter_by(user_id=user.id)\
            .order_by(Attendance.timestamp.desc())\
            .paginate(page=page, per_page=per_page, error_out=False)
        
        return jsonify({
            'records': [record.to_dict() for record in attendance_records.items],
            'total': attendance_records.total,
            'page': page,
            'pages': attendance_records.pages
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching attendance history: {e}")
        return jsonify({'error': 'Failed to fetch attendance history'}), 500


# ===================== ADMIN DASHBOARD =====================

@app.route('/admin')
@jwt_required(optional=True)
def admin_dashboard():
    """Admin dashboard page"""
    return render_template('admin_dashboard.html')


@app.route('/api/admin/stats', methods=['GET'])
@role_required('admin', 'super_admin')
def admin_stats():
    """Get admin statistics"""
    try:
        total_users = User.query.filter_by(role='user').count()
        active_users = User.query.filter_by(role='user', status='active').count()
        pending_requests = EnrollmentRequest.query.filter_by(status='pending').count()
        
        today = datetime.utcnow().date()
        today_attendance = Attendance.query.filter(Attendance.date == today).count()
        
        # Get recent attendance
        recent_attendance = Attendance.query.order_by(Attendance.timestamp.desc()).limit(10).all()
        
        return jsonify({
            'total_users': total_users,
            'active_users': active_users,
            'pending_requests': pending_requests,
            'today_attendance': today_attendance,
            'recent_attendance': [a.to_dict() for a in recent_attendance]
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching admin stats: {e}")
        return jsonify({'error': 'Failed to fetch statistics'}), 500


@app.route('/api/admin/enrollment-requests', methods=['GET'])
@role_required('admin', 'super_admin')
def get_enrollment_requests():
    """Get all enrollment requests"""
    try:
        status = request.args.get('status', 'pending')
        requests = EnrollmentRequest.query.filter_by(status=status)\
            .order_by(EnrollmentRequest.submitted_at.desc()).all()
        
        return jsonify({
            'requests': [req.to_dict(include_images=True) for req in requests]
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching enrollment requests: {e}")
        return jsonify({'error': 'Failed to fetch requests'}), 500


@app.route('/api/admin/enrollment-requests/<int:request_id>/approve', methods=['POST'])
@role_required('admin', 'super_admin')
def approve_enrollment(request_id):
    """Approve enrollment request"""
    try:
        current_user_email = get_jwt_identity()
        admin = User.query.filter_by(email=current_user_email).first()
        
        enroll_req = EnrollmentRequest.query.get(request_id)
        if not enroll_req:
            return jsonify({'error': 'Request not found'}), 404
        
        if enroll_req.status != 'pending':
            return jsonify({'error': 'Request already processed'}), 400
        
        # Extract face embeddings
        try:
            embedding = face_service.extract_multiple_embeddings(enroll_req.images)
        except Exception as e:
            return jsonify({'error': f'Failed to process images: {str(e)}'}), 400
        
        # Create user
        user = User(
            name=enroll_req.name,
            email=enroll_req.email,
            phone=enroll_req.phone,
            department=enroll_req.department,
            password_hash=enroll_req.password_hash,
            role='user',
            status='active'
        )
        db.session.add(user)
        db.session.flush()
        
        # Create person record
        person = Person(
            user_id=user.id,
            name=enroll_req.name,
            embedding=embedding.tobytes(),
            embedding_dim=len(embedding),
            photos_count=len(enroll_req.images),
            status='active',
            enrolled_by=admin.email
        )
        db.session.add(person)
        db.session.flush()
        
        # Add to FAISS index
        faiss_service.add_person(person.id, person.name, embedding)
        
        # Update request status
        enroll_req.status = 'approved'
        enroll_req.processed_at = datetime.utcnow()
        enroll_req.processed_by = admin.email
        
        db.session.commit()
        
        log_activity('enrollment_approved', admin.id, 'enrollment_request', request_id, 
                    {'user_id': user.id, 'person_id': person.id})
        
        return jsonify({
            'message': 'Enrollment approved successfully',
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error approving enrollment: {e}")
        return jsonify({'error': 'Failed to approve enrollment'}), 500


@app.route('/api/admin/enrollment-requests/<int:request_id>/reject', methods=['POST'])
@role_required('admin', 'super_admin')
def reject_enrollment(request_id):
    """Reject enrollment request"""
    try:
        current_user_email = get_jwt_identity()
        admin = User.query.filter_by(email=current_user_email).first()
        
        data = request.get_json()
        reason = data.get('reason', 'No reason provided')
        
        enroll_req = EnrollmentRequest.query.get(request_id)
        if not enroll_req:
            return jsonify({'error': 'Request not found'}), 404
        
        if enroll_req.status != 'pending':
            return jsonify({'error': 'Request already processed'}), 400
        
        enroll_req.status = 'rejected'
        enroll_req.processed_at = datetime.utcnow()
        enroll_req.processed_by = admin.email
        enroll_req.rejection_reason = reason
        
        db.session.commit()
        
        log_activity('enrollment_rejected', admin.id, 'enrollment_request', request_id)
        
        return jsonify({'message': 'Enrollment rejected'}), 200
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error rejecting enrollment: {e}")
        return jsonify({'error': 'Failed to reject enrollment'}), 500


# ===================== ATTENDANCE ROUTES =====================

@app.route('/mark-attendance')
def mark_attendance_page():
    """Mark attendance page"""
    return render_template('mark_attendance.html')


@app.route('/api/attendance/mark', methods=['POST'])
def mark_attendance():
    """Mark attendance using face recognition"""
    try:
        data = request.get_json()
        image = data.get('image')
        
        if not image:
            return jsonify({'error': 'Image required'}), 400
        
        # Extract embedding from image
        try:
            embedding = face_service.extract_embedding(image)
        except Exception as e:
            return jsonify({'error': f'Face detection failed: {str(e)}'}), 400
        
        # Search for matching person
        matches = faiss_service.search(embedding, k=1, threshold=0.6)
        
        if not matches:
            return jsonify({'error': 'No matching person found'}), 404
        
        match = matches[0]
        person = Person.query.get(match['person_id'])
        
        if not person:
            return jsonify({'error': 'Person record not found'}), 404
        
        # Check if already marked today
        today = datetime.utcnow().date()
        existing = Attendance.query.filter(
            Attendance.person_id == person.id,
            Attendance.date == today
        ).first()
        
        if existing:
            return jsonify({
                'message': 'Attendance already marked today',
                'attendance': existing.to_dict()
            }), 200
        
        # Create attendance record
        attendance = Attendance(
            user_id=person.user_id,
            person_id=person.id,
            name=person.name,
            confidence=match['confidence'],
            date=today,
            status='present'
        )
        
        db.session.add(attendance)
        db.session.commit()
        
        log_activity('attendance_marked', person.user_id, 'attendance', attendance.id)
        
        return jsonify({
            'message': 'Attendance marked successfully',
            'attendance': attendance.to_dict(),
            'person': person.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error marking attendance: {e}")
        return jsonify({'error': 'Failed to mark attendance'}), 500


# ===================== SUPER ADMIN ROUTES =====================

@app.route('/super-admin')
@jwt_required(optional=True)
def super_admin_dashboard():
    """Super admin dashboard"""
    return render_template('super_admin_dashboard.html')


@app.route('/api/super-admin/users', methods=['GET'])
@role_required('super_admin')
def get_all_users():
    """Get all users (super admin only)"""
    try:
        users = User.query.order_by(User.created_at.desc()).all()
        return jsonify({
            'users': [user.to_dict() for user in users]
        }), 200
    except Exception as e:
        logger.error(f"Error fetching users: {e}")
        return jsonify({'error': 'Failed to fetch users'}), 500


@app.route('/api/super-admin/users/<int:user_id>', methods=['PUT'])
@role_required('super_admin')
def update_user(user_id):
    """Update user details"""
    try:
        current_user_email = get_jwt_identity()
        admin = User.query.filter_by(email=current_user_email).first()
        
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        data = request.get_json()
        
        if 'name' in data:
            user.name = data['name']
        if 'email' in data:
            user.email = data['email']
        if 'role' in data:
            user.role = data['role']
        if 'status' in data:
            user.status = data['status']
        if 'department' in data:
            user.department = data['department']
        if 'phone' in data:
            user.phone = data['phone']
        
        db.session.commit()
        
        log_activity('user_updated', admin.id, 'user', user_id)
        
        return jsonify({
            'message': 'User updated successfully',
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error updating user: {e}")
        return jsonify({'error': 'Failed to update user'}), 500


@app.route('/api/super-admin/users/<int:user_id>', methods=['DELETE'])
@role_required('super_admin')
def delete_user(user_id):
    """Delete user"""
    try:
        current_user_email = get_jwt_identity()
        admin = User.query.filter_by(email=current_user_email).first()
        
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        if user.id == admin.id:
            return jsonify({'error': 'Cannot delete yourself'}), 400
        
        # Delete related records
        Person.query.filter_by(user_id=user_id).delete()
        Attendance.query.filter_by(user_id=user_id).delete()
        SystemLog.query.filter_by(user_id=user_id).delete()
        
        db.session.delete(user)
        db.session.commit()
        
        log_activity('user_deleted', admin.id, 'user', user_id)
        
        return jsonify({'message': 'User deleted successfully'}), 200
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting user: {e}")
        return jsonify({'error': 'Failed to delete user'}), 500


@app.route('/api/super-admin/logs', methods=['GET'])
@role_required('super_admin')
def get_system_logs():
    """Get system activity logs"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        
        logs = SystemLog.query.order_by(SystemLog.timestamp.desc())\
            .paginate(page=page, per_page=per_page, error_out=False)
        
        return jsonify({
            'logs': [log.to_dict() for log in logs.items],
            'total': logs.total,
            'page': page,
            'pages': logs.pages
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching logs: {e}")
        return jsonify({'error': 'Failed to fetch logs'}), 500


@app.route('/api/super-admin/stats', methods=['GET'])
@role_required('super_admin')
def super_admin_stats():
    """Get comprehensive system statistics"""
    try:
        total_users = User.query.count()
        total_admins = User.query.filter_by(role='admin').count()
        total_super_admins = User.query.filter_by(role='super_admin').count()
        active_users = User.query.filter_by(status='active').count()
        inactive_users = User.query.filter_by(status='inactive').count()
        suspended_users = User.query.filter_by(status='suspended').count()
        
        total_persons = Person.query.count()
        total_attendance = Attendance.query.count()
        total_requests = EnrollmentRequest.query.count()
        pending_requests = EnrollmentRequest.query.filter_by(status='pending').count()
        
        today = datetime.utcnow().date()
        today_attendance = Attendance.query.filter(Attendance.date == today).count()
        
        return jsonify({
            'users': {
                'total': total_users,
                'admins': total_admins,
                'super_admins': total_super_admins,
                'active': active_users,
                'inactive': inactive_users,
                'suspended': suspended_users
            },
            'persons': total_persons,
            'attendance': {
                'total': total_attendance,
                'today': today_attendance
            },
            'enrollment_requests': {
                'total': total_requests,
                'pending': pending_requests
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        return jsonify({'error': 'Failed to fetch statistics'}), 500


# ===================== HEALTH CHECK =====================

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        # Check database
        db.session.execute(db.text('SELECT 1'))
        
        # Check FAISS
        total_persons = faiss_service.get_total_persons()
        
        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'faiss_persons': total_persons
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503


# ===================== INITIALIZATION =====================

def init_database():
    """Initialize database with default data"""
    with app.app_context():
        db.create_all()
        
        # Create super admin if not exists
        super_admin_email = os.getenv('SUPER_ADMIN_EMAIL', 'superadmin@attendance.com')
        super_admin_password = os.getenv('SUPER_ADMIN_PASSWORD', 'SuperAdmin@123')
        
        if not User.query.filter_by(email=super_admin_email).first():
            super_admin = User(
                name='Super Admin',
                email=super_admin_email,
                role='super_admin',
                status='active'
            )
            super_admin.set_password(super_admin_password)
            db.session.add(super_admin)
            db.session.commit()
            logger.info(f"Super admin created: {super_admin_email}")
        
        # Rebuild FAISS index
        persons = Person.query.filter_by(status='active').all()
        if persons:
            faiss_service.rebuild_from_database([
                (p.id, p.name, p.embedding) for p in persons
            ])
            logger.info(f"FAISS index rebuilt with {len(persons)} persons")


if __name__ == '__main__':
    init_database()
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
