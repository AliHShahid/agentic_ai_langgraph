from django.db import models


class Lead(models.Model):
    PACKAGE_CHOICES = [
        ('chatbots', 'AI Chatbots & RAG'),
        ('automation', 'Workflow Automation'),
        ('mobile', 'Mobile App Development'),
        ('consult', 'AI Consultation'),
        ('web', 'Web Development'),
        ('analytics', 'Data Analysis'),
        ('mldl', 'Machine Learning Tasks'),
        ('custom', 'Custom Request'),
    ]

    SUBMITTED_BY_CHOICES = [
        ('web', 'Website Form'),
        ('chatbot', 'AI Chatbot'),
    ]

    name = models.CharField(max_length=255)
    email = models.EmailField()
    company = models.CharField(max_length=255)
    package = models.CharField(max_length=50, choices=PACKAGE_CHOICES)
    phone = models.CharField(max_length=20)
    message = models.TextField()
    submitted_by = models.CharField(max_length=50, default='web', choices=SUBMITTED_BY_CHOICES)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} - {self.package}"


ContactLead = Lead

