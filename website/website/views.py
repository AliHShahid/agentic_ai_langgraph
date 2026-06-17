import json

from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .models import ContactLead, Lead


def _get_payload(data_source):
    return {
        'name': (data_source.get('name') or '').strip(),
        'email': (data_source.get('email') or '').strip(),
        'company': (data_source.get('company') or '').strip(),
        'package': (data_source.get('package') or '').strip(),
        'phone': (data_source.get('phone') or '').strip(),
        'message': (data_source.get('message') or '').strip(),
    }


def _validate_payload(payload):
    required_fields = ['name', 'email', 'company', 'package', 'phone']
    missing_fields = [field for field in required_fields if not payload.get(field)]
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"

    allowed_packages = {choice[0] for choice in Lead.PACKAGE_CHOICES}
    if payload['package'] not in allowed_packages:
        return False, 'Invalid package selection'

    return True, ''


def _save_lead(payload, submitted_by='web'):
    return ContactLead.objects.create(
        name=payload['name'],
        email=payload['email'],
        company=payload['company'],
        package=payload['package'],
        phone=payload['phone'],
        message=payload['message'],
        submitted_by=submitted_by,
    )


def _send_confirmation_email(lead):
    from django.conf import settings
    from django.core.mail import send_mail

    subject = 'We received your inquiry - DEVRYZE'
    message = (
        f"Hi {lead.name},\n\n"
        'Thank you for reaching out to DEVRYZE!\n\n'
        f"We received your inquiry about: {lead.get_package_display()}\n\n"
        f"Our team will review your request and contact you shortly at {lead.phone} or {lead.email}.\n\n"
        'Best regards,\n'
        'DEVRYZE Team'
    )

    from_email = getattr(settings, 'DEFAULT_FROM_EMAIL', 'noreply@devryze.tech')

    try:
        send_mail(subject, message, from_email, [lead.email], fail_silently=False)
    except Exception:
        pass


def _submit_lead(data_source, submitted_by='web'):
    payload = _get_payload(data_source)
    is_valid, error_message = _validate_payload(payload)
    if not is_valid:
        return None, error_message

    lead = _save_lead(payload, submitted_by=submitted_by)
    _send_confirmation_email(lead)
    return lead, ''


@require_http_methods(['GET', 'POST'])
def home(request):
    if request.method == 'POST':
        content_type = request.headers.get('content-type', '')
        if content_type.startswith('application/json'):
            return handle_chatbot_submission(request)

        lead, error_message = _submit_lead(request.POST, submitted_by='web')
        if lead is None:
            messages.error(request, error_message)
            return redirect('home')

        messages.success(request, 'Your message has been sent successfully!')
        return redirect('home')

    return render(request, 'index.html')


def how_it_works(request):
    return render(request, 'how_it_works.html')

def about(request):
    return render(request, 'about.html')

def case_study(request):
    return render(request, 'case_study.html')

def pricing(request):
    return render(request, 'pricing.html')

def services(request):
    return render(request, 'services.html')

def faq(request):
    return render(request, 'faq.html')

def saamay(request):
    return render(request, 'saamay.html')

def adryze(request):
    return render(request, 'adryze.html')

def documind(request):
    return render(request, 'documind.html')


def contact_view(request):
    return home(request)


@require_http_methods(['POST'])
@csrf_exempt
def handle_chatbot_submission(request):
    try:
        if request.content_type and request.content_type.startswith('application/json'):
            data = json.loads(request.body or b'{}')
        else:
            data = request.POST
    except json.JSONDecodeError:
        return JsonResponse({'status': 'error', 'message': 'Invalid JSON'}, status=400)

    lead, error_message = _submit_lead(data, submitted_by='chatbot')
    if lead is None:
        return JsonResponse({'status': 'error', 'message': error_message}, status=400)

    return JsonResponse(
        {
            'status': 'success',
            'message': f'Thank you, {lead.name}! We received your inquiry.',
            'lead_id': lead.id,
        },
        status=201,
    )
